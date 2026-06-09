#!/usr/bin/env python3
"""Aggregate scores/*.json (+ run latency) into a side-by-side scorecard."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the bake-off scorecard")
    ap.add_argument("--config", default=str(ROOT / "bakeoff" / "config.json"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    scores_dir = ROOT / cfg["paths"]["scores"]
    runs_dir = ROOT / cfg["paths"]["runs"]
    files = sorted(scores_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"report: no scores in {scores_dir} — run score.py first")

    header = ["model", "field_acc", "field_prec", "empty_clean",
              "halluc", "value_recall", "avg_latency_s"]
    rows = []
    for f in files:
        if f.name == "scorecard.json":
            continue
        d = json.loads(f.read_text())
        m, model = d["metrics"], d["model"]
        rows.append([
            model,
            m["field_gold"]["accuracy"],
            m["field_gold"]["precision"],
            m["empty_fidelity"]["clean_rate"],
            m["empty_fidelity"]["hallucination_rate"],
            m["value_recall"]["recall"],
            latency_stats(runs_dir / model, cfg["pages"]),
        ])

    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    def fmt(r):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
    lines = [fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows]
    print("\n".join(lines))

    md = "# Bake-off scorecard\n\n"
    md += "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for r in rows:
        md += "| " + " | ".join(str(c) for c in r) + " |\n"
    (scores_dir / "scorecard.md").write_text(md, encoding="utf-8")
    print(f"\nok: wrote {scores_dir / 'scorecard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
