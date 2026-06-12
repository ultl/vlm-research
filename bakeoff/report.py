#!/usr/bin/env python3
"""Aggregate scores/*.json into one scorecard across tracks.

Handles three score schemas: Track A (run.py/score.py — whole-doc parse),
Track B-kie (kie.py — single-VLM field extraction), Track B-rag (chatocr.py —
PP-ChatOCRv4 RAG-KIE). One row per (model, track).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def latency_stats(run_dir: Path, pages) -> Optional[float]:
    lats = []
    for name in pages:
        mp = run_dir / f"{name}.meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            if m.get("status") == "ok":
                lats.append(m["latency_s"])
    return round(statistics.mean(lats), 2) if lats else None


def _r(x):
    return "-" if x is None else x


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the bake-off scorecard")
    ap.add_argument("--config", default=str(ROOT / "bakeoff" / "config.json"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    scores_dir = ROOT / cfg["paths"]["scores"]
    runs_dir = ROOT / cfg["paths"]["runs"]
    files = sorted(f for f in scores_dir.glob("*.json") if f.name != "scorecard.json")
    if not files:
        raise SystemExit(f"report: no scores in {scores_dir} — run a model first")

    header = ["model", "track", "field_acc", "field_prec",
              "empty_clean", "halluc", "value_recall", "latency_s"]
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        m, model = d["metrics"], d["model"]
        pages = d.get("pages", cfg["pages"])
        if "field_gold" in m:                                  # Track A (parse)
            rows.append([
                model, "A:parse",
                _r(m["field_gold"]["accuracy"]), _r(m["field_gold"]["precision"]),
                _r(m["empty_fidelity"]["clean_rate"]),
                _r(m["empty_fidelity"]["hallucination_rate"]),
                _r(m["value_recall"]["recall"]),
                _r(latency_stats(runs_dir / model, pages)),
            ])
        else:                                                  # Track B (kie / rag)
            task = d.get("task", "kie")
            halluc = (round(m["hallucinated"] / m["empty_n"], 3)
                      if m.get("empty_n") else None)
            rows.append([
                model, "B:kie" if task == "kie" else "B:rag",
                _r(m.get("field_accuracy")), "-",
                _r(m.get("empty_clean_rate")), _r(halluc), "-",
                _r(m.get("latency_s") or latency_stats(runs_dir / f"{model}__kie", pages)),
            ])

    rows.sort(key=lambda r: (r[0], r[1]))
    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    fmt = lambda r: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    print("\n".join([fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows]))

    md = "# Bake-off scorecard\n\n| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for r in rows:
        md += "| " + " | ".join(str(c) for c in r) + " |\n"
    (scores_dir / "scorecard.md").write_text(md, encoding="utf-8")
    print(f"\nok: wrote {scores_dir / 'scorecard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
