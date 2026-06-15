#!/usr/bin/env python3
"""Run one model over the page images -> runs/<model>/ (the output contract).

This is the GPU-side entrypoint. Adapter modules are imported lazily by name so
the heavy ML deps only load for the model actually selected.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `models` importable

from page_selection import resolve_page_names  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a VLM over the page images")
    ap.add_argument("--config", default=str(ROOT / "bakeoff" / "config.json"))
    ap.add_argument("--model", required=True, help="model key in config.models")
    ap.add_argument(
        "--pages",
        nargs="+",
        help="page names to run, e.g. page1 page3 or page1,page3",
    )
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.model not in cfg["models"]:
        sys.exit(f"run: unknown model '{args.model}'; "
                 f"known: {', '.join(cfg['models'])}")
    mcfg = cfg["models"][args.model]

    prompt_ref = mcfg.get("prompt_path", cfg["paths"]["prompt"])
    prompt_path = Path(prompt_ref)
    if not prompt_path.is_absolute():
        prompt_path = ROOT / prompt_path
    if not prompt_path.exists():
        sys.exit(f"run: prompt file missing: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    try:
        page_names = resolve_page_names(cfg["pages"], args.pages)
    except ValueError as exc:
        sys.exit(f"run: {exc}")

    print(
        f"run: model={args.model} adapter={mcfg['adapter']} "
        f"pages={','.join(page_names)} prompt={prompt_path}",
        flush=True,
    )

    from models.base import Runner

    mod = importlib.import_module(f"models.{mcfg['adapter']}")
    adapter = mod.build(mcfg)

    Runner(ROOT).execute(
        adapter,
        model_id=mcfg["model_id"],
        decode=mcfg.get("decode", {}),
        pages_dir=ROOT / cfg["paths"]["pages"],
        page_names=page_names,
        prompt=prompt,
        out_dir=ROOT / cfg["paths"]["runs"] / args.model,
        pdf_path=ROOT / cfg["paths"]["pdf"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
