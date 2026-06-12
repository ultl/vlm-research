#!/usr/bin/env python3
"""Track B — per-page field extraction (KIE) for prompt-using VLMs.

For each page: build a field-list prompt from the gold, ask the model for a JSON
object {field_id: value|null}, and score by DIRECT field match (no markdown
alignment). Keeps the empty-fidelity test (blank field -> null).

VLM-only: pipeline parsers (uses_prompt=false) are prompt-driven but only via a
closed, auto-selected task vocabulary (no free-text hook), so they can't take a
field-list instruction. Writes runs/<model>__kie/ and scores/<model>__kie.json.
Runs + scores in one pass.

Usage: python bakeoff/kie.py --model qwen25vl
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from page_selection import resolve_page_names
from score import to_int, flatten_gold, bucket  # reuse normalization + bucketing

ROOT = Path(__file__).resolve().parent.parent

KIE_HEADER = (
    "You are reading ONE page of a Japanese tax form. Extract the value of each "
    "listed field from THIS page.\n"
    "Output ONLY a JSON object mapping each field id (string) to its value:\n"
    "- an integer with NO commas or currency symbol for money amounts,\n"
    "- null if the field is blank/empty on the page.\n"
    "Do not guess or invent values — use null for blanks.\n\n"
    "Fields:\n"
)


def page_fields(gold_cells):
    """The id'd, scoreable cells for a page (filled GOLD/CANDIDATE + empty)."""
    out = []
    for c in gold_cells:
        if c.get("id") and bucket(c) in ("gold", "candidate", "gold_empty"):
            out.append(c)
    return out


def build_prompt(fields):
    lines = [f'- "{c["id"]}": {c.get("label") or ""}' for c in fields]
    return KIE_HEADER + "\n".join(lines) + "\n\nJSON:"


def parse_json(text):
    """Extract the first {...} block from the model output and parse it."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def is_blank(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in ("", "null", "none", "n/a", "-")
    return False


def score_page(pred, fields):
    rows = []
    for c in fields:
        b, pid = bucket(c), c["id"]
        present = pid in pred
        raw = pred.get(pid)
        if b in ("gold", "candidate"):
            gi = to_int(c["value"])
            pv = to_int(raw) if present else None
            verdict = ("missed" if pv is None else
                       "correct" if pv == gi else "wrong")
        else:  # gold_empty: correct == model returned blank/null
            verdict = "clean" if (not present or is_blank(raw)) else "hallucinated"
        rows.append({"id": pid, "bucket": b, "gold": c["value"],
                     "pred": raw if present else None, "verdict": verdict})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Track B: field-extraction (KIE)")
    ap.add_argument("--config", default=str(ROOT / "bakeoff" / "config.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--pages",
        nargs="+",
        help="page names to run, e.g. page1 page3 or page1,page3",
    )
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    try:
        page_names = resolve_page_names(cfg["pages"], args.pages)
    except ValueError as exc:
        sys.exit(f"kie: {exc}")
    if args.model not in cfg["models"]:
        sys.exit(f"kie: unknown model '{args.model}'")
    m = cfg["models"][args.model]
    if not (m.get("uses_prompt") and m.get("accepts") == "images"):
        sys.exit(f"kie: '{args.model}' has no free-text prompt hook (closed task "
                 "vocabulary) — Track B needs an instruction-following image VLM")

    adapter = importlib.import_module(f"models.{m['adapter']}").build(m)
    print(f"[kie:{args.model}] loading model…", flush=True)
    adapter.load()
    print(f"[kie:{args.model}] model loaded", flush=True)

    gold_dir = ROOT / cfg["paths"]["gold"]
    out_dir = ROOT / cfg["paths"]["runs"] / f"{args.model}__kie"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name in page_names:
        cells = flatten_gold(json.loads((gold_dir / f"{name}.cells.json").read_text()))
        fields = page_fields(cells)
        prompt = build_prompt(fields)
        img = ROOT / cfg["paths"]["pages"] / f"{name}.png"
        print(f"[kie:{args.model}] {name} ({len(fields)} fields) inferring…", flush=True)
        t0 = time.perf_counter()
        res = adapter.infer(str(img), prompt)
        dt = time.perf_counter() - t0
        (out_dir / f"{name}.json").write_text(res.text, encoding="utf-8")
        rows = score_page(parse_json(res.text), fields)
        for r in rows:
            r["page"] = name
        all_rows += rows
        ok = sum(1 for r in rows if r["verdict"] in ("correct", "clean"))
        print(f"[kie:{args.model}] {name} {ok}/{len(rows)} ok in {dt:.1f}s", flush=True)

    filled = [r for r in all_rows if r["bucket"] in ("gold", "candidate")]
    empt = [r for r in all_rows if r["bucket"] == "gold_empty"]

    def rate(rows, good):
        return round(sum(1 for r in rows if r["verdict"] in good) / len(rows), 3) if rows else None

    metrics = {
        "field_accuracy": rate(filled, ("correct",)),
        "field_n": len(filled),
        "field_correct": sum(1 for r in filled if r["verdict"] == "correct"),
        "empty_clean_rate": rate(empt, ("clean",)),
        "empty_n": len(empt),
        "hallucinated": sum(1 for r in empt if r["verdict"] == "hallucinated"),
    }
    result = {
        "model": args.model,
        "task": "kie",
        "pages": page_names,
        "metrics": metrics,
        "fields": all_rows,
    }
    sdir = ROOT / cfg["paths"]["scores"]
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"{args.model}__kie.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ok: kie[{args.model}] field_acc {metrics['field_accuracy']} "
          f"({metrics['field_correct']}/{metrics['field_n']}) | "
          f"empty_clean {metrics['empty_clean_rate']} "
          f"(halluc {metrics['hallucinated']}/{metrics['empty_n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
