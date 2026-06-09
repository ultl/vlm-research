#!/usr/bin/env python3
"""Score a model's run against the gold cells (CPU, deterministic, no LLM judge).

Pipeline: normalize -> align (label/number-anchored, with a value-presence
cross-check) -> per-field verdict -> the three metrics in scored_mask.md.

Alignment from free markdown is heuristic and the main source of scoring error;
`value_recall` (extraction-free) and the `unmatched` list exist to audit it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


# ---------- normalization (mirrors fixtures/gt/schema.md) ----------

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def norm_text(s: str) -> str:
    return nfkc(s).lower().strip()


def to_int(value) -> Optional[int]:
    """Normalize a value to an integer yen amount, or None if non-numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    t = nfkc(value).replace(",", "").replace("¥", "").replace("円", "").strip()
    return int(t) if re.fullmatch(r"\d+", t) else None


def line_numbers(line: str) -> List[int]:
    """Integers in a line after NFKC + comma stripping."""
    t = nfkc(line)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    return [int(x) for x in re.findall(r"\d+", t)]


# ---------- gold flattening + bucketing ----------

def flatten_gold(cells: dict) -> List[dict]:
    """Yield scoreable cells: {label, no, value, status, conf}."""
    out: List[dict] = []

    def visit(o):
        if isinstance(o, dict):
            if "status" in o and ("value" in o or "label_ja" in o or "category" in o):
                out.append({
                    "label": o.get("label_ja") or o.get("category"),
                    "no": o.get("no"),
                    "value": o.get("value"),
                    "status": o.get("status"),
                    "conf": o.get("conf"),
                })
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)

    visit(cells)
    return out


def bucket(cell: dict) -> str:
    status, conf, val = cell["status"], cell.get("conf"), cell["value"]
    if status == "filled" and conf == "H" and to_int(val) is not None:
        return "gold"
    if status == "filled" and conf == "M" and to_int(val) is not None:
        return "candidate"
    if status == "empty" and cell.get("label"):
        return "gold_empty"
    return "excluded"


# ---------- alignment ----------

def find_anchor_line(lines: List[str], cell: dict) -> Optional[str]:
    """First line matching the field's label (fuzzy) or its numeric `no`."""
    label = norm_text(cell["label"]) if cell.get("label") else None
    no = str(cell["no"]) if cell.get("no") else None
    no_int = to_int(no) if no else None
    for ln in lines:
        nl = norm_text(ln)
        if label and len(label) >= 2 and label in nl:
            return ln
        if no_int is not None and no_int in line_numbers(ln) and any(
                c.isalpha() or ord(c) > 0x3000 for c in ln):
            return ln
    return None


def extract_value(line: str, cell: dict) -> Optional[int]:
    """Pick the field's value from its anchor line (largest non-`no` number)."""
    nums = line_numbers(line)
    no_int = to_int(str(cell["no"])) if cell.get("no") else None
    cands = [n for n in nums if n != no_int]
    if not cands:
        return None
    return max(cands, key=lambda n: (len(str(n)), n))  # amounts are the big number


# ---------- scoring ----------

def score_page(md: str, cells: List[dict]) -> dict:
    lines = [ln for ln in md.splitlines() if ln.strip()]
    text_nums = set()
    for ln in lines:
        text_nums.update(n for n in line_numbers(ln) if n >= 10)

    rows = []
    for c in cells:
        b = bucket(c)
        if b == "excluded":
            continue
        rec = {"label": c["label"], "no": c.get("no"), "bucket": b,
               "gold": c["value"]}
        anchor = find_anchor_line(lines, c)
        if b in ("gold", "candidate"):
            gold_int = to_int(c["value"])
            ext = extract_value(anchor, c) if anchor else None
            if ext is None:
                rec["verdict"] = "missed"
            elif ext == gold_int:
                rec["verdict"] = "correct"
            else:
                rec["verdict"] = "wrong"
            rec["extracted"] = ext
            rec["present_anywhere"] = gold_int in text_nums
        elif b == "gold_empty":
            if anchor and extract_value(anchor, c) is not None:
                rec["verdict"] = "hallucinated"
            else:
                rec["verdict"] = "clean"
        rows.append(rec)
    return {"rows": rows, "text_nums": text_nums}


def aggregate(all_rows: List[dict], text_nums_all: set, gold_values: set) -> dict:
    def counts(rows, verdicts):
        return {v: sum(1 for r in rows if r.get("verdict") == v) for v in verdicts}

    gold = [r for r in all_rows if r["bucket"] == "gold"]
    cand = [r for r in all_rows if r["bucket"] == "candidate"]
    empty = [r for r in all_rows if r["bucket"] == "gold_empty"]

    gc = counts(gold, ["correct", "wrong", "missed"])
    n_gold = sum(gc.values())
    answered = gc["correct"] + gc["wrong"]
    cc = counts(cand, ["correct", "wrong", "missed"])
    n_cand = sum(cc.values())
    ec = counts(empty, ["clean", "hallucinated"])
    n_empty = sum(ec.values())

    return {
        "field_gold": {
            **gc, "n": n_gold,
            "accuracy": round(gc["correct"] / n_gold, 3) if n_gold else None,
            "precision": round(gc["correct"] / answered, 3) if answered else None,
        },
        "field_candidate_provisional": {
            **cc, "n": n_cand,
            "accuracy": round(cc["correct"] / n_cand, 3) if n_cand else None,
        },
        "empty_fidelity": {
            **ec, "n": n_empty,
            "clean_rate": round(ec["clean"] / n_empty, 3) if n_empty else None,
            "hallucination_rate": round(ec["hallucinated"] / n_empty, 3) if n_empty else None,
        },
        "value_recall": {
            "present": len(gold_values & text_nums_all), "total": len(gold_values),
            "recall": round(len(gold_values & text_nums_all) / len(gold_values), 3)
            if gold_values else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a model run against gold")
    ap.add_argument("--config", default=str(ROOT / "bakeoff" / "config.json"))
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    gold_dir = ROOT / cfg["paths"]["gold"]
    run_dir = ROOT / cfg["paths"]["runs"] / args.model
    if not run_dir.exists():
        sys.exit(f"score: no run found for '{args.model}' at {run_dir}")

    all_rows: List[dict] = []
    text_nums_all: set = set()
    gold_values: set = set()
    per_page = {}

    for name in cfg["pages"]:
        gold_path = gold_dir / f"{name}.cells.json"
        md_path = run_dir / f"{name}.md"
        if not gold_path.exists():
            sys.exit(f"score: gold missing: {gold_path}")
        if not md_path.exists():
            sys.exit(f"score: model output missing: {md_path}")

        cells = flatten_gold(json.loads(gold_path.read_text()))
        for c in cells:
            if bucket(c) in ("gold", "candidate"):
                gi = to_int(c["value"])
                if gi is not None and gi >= 10:
                    gold_values.add(gi)

        page = score_page(md_path.read_text(encoding="utf-8"), cells)
        for r in page["rows"]:
            r["page"] = name
        all_rows.extend(page["rows"])
        text_nums_all |= page["text_nums"]
        per_page[name] = page["rows"]

    agg = aggregate(all_rows, text_nums_all, gold_values)
    result = {"model": args.model, "metrics": agg,
              "fields": all_rows,
              "unmatched": [r for r in all_rows if r.get("verdict") == "missed"]}

    out_dir = ROOT / cfg["paths"]["scores"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.model}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    m = agg
    print(f"ok: scored {args.model} -> {out_dir / (args.model + '.json')}")
    print(f"  field acc {m['field_gold']['accuracy']} "
          f"({m['field_gold']['correct']}/{m['field_gold']['n']}) | "
          f"empty clean {m['empty_fidelity']['clean_rate']} | "
          f"value-recall {m['value_recall']['recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
