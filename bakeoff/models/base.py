"""Adapter interface + the shared runner.

Adapters stay thin: declare capabilities, load() the model, infer() one page.
The Runner owns everything uniform — timing, metadata, provenance, error
capture, and writing the output contract (pageN.md / pageN.meta.json / run.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class InferResult:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    ttft_s: Optional[float] = None


class Adapter(ABC):
    name: str = "base"
    accepts: str = "images"          # "images" | "pdf"
    uses_prompt: bool = True
    exposes_tokens: bool = True

    @abstractmethod
    def load(self) -> None:
        """Pull weights + put the model on the GPU. Excluded from latency."""

    @abstractmethod
    def infer(self, image_path: str, prompt: str) -> InferResult:
        """Run one page image. Pipeline (pdf) adapters override the runner path."""

    def env_info(self) -> Dict[str, object]:
        """GPU / CUDA / model revision — best-effort, recorded in run.json."""
        return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def code_sha(root: Path) -> str:
    if os.environ.get("BAKEOFF_CODE_SHA"):
        return os.environ["BAKEOFF_CODE_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Runner:
    def __init__(self, root: Path):
        self.root = root

    def execute(self, adapter: Adapter, *, model_id: str, decode: dict,
                pages_dir: Path, page_names: List[str], prompt: str,
                out_dir: Path, pdf_path: Path) -> dict:
        if adapter.accepts != "images":
            raise NotImplementedError(
                f"runner supports accepts='images'; '{adapter.accepts}' "
                "lands with the pipeline (MinerU/PaddleOCR-VL) adapters")

        out_dir.mkdir(parents=True, exist_ok=True)
        started = _now()
        adapter.load()

        page_records: List[dict] = []
        errors = 0
        total_latency = 0.0

        for name in page_names:
            img = pages_dir / f"{name}.png"
            if not img.exists():
                raise SystemExit(f"run: page image missing: {img}")
            w, h = Image.open(img).size

            status, err, text = "ok", None, ""
            res = InferResult(text="")
            t0 = time.perf_counter()
            try:
                res = adapter.infer(str(img), prompt)
                text = res.text
            except Exception:
                status, err = "error", traceback.format_exc()
                errors += 1
            latency = time.perf_counter() - t0
            total_latency += latency

            (out_dir / f"{name}.md").write_text(text, encoding="utf-8")
            tps = (res.output_tokens / latency
                   if res.output_tokens and latency > 0 else None)
            meta = {
                "page": name,
                "input_image": str(img.relative_to(self.root)),
                "input_image_sha256": sha256_file(img),
                "image_wh": [w, h],
                "latency_s": round(latency, 3),
                "ttft_s": res.ttft_s,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "tokens_per_s": round(tps, 2) if tps else None,
                "output_chars": len(text),
                "decode": decode,
                "status": status,
                "error": err,
            }
            (out_dir / f"{name}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            page_records.append({"page": name, "status": status,
                                 "latency_s": round(latency, 3)})

        run = {
            "model": adapter.name,
            "model_id": model_id,
            "adapter": adapter.name,
            "accepts": adapter.accepts,
            "uses_prompt": adapter.uses_prompt,
            "exposes_tokens": adapter.exposes_tokens,
            "env": adapter.env_info(),
            "image_digest": os.environ.get("BAKEOFF_IMAGE_DIGEST", "unknown"),
            "code_sha": code_sha(self.root),
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "input_pdf_sha256": sha256_file(pdf_path),
            "decode": decode,
            "started_at": started,
            "finished_at": _now(),
            "totals": {"pages": len(page_names), "errors": errors,
                       "total_latency_s": round(total_latency, 3)},
            "pages": page_records,
        }
        (out_dir / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

        if errors:
            print(f"run[{adapter.name}]: completed with {errors} error(s) "
                  f"-> {out_dir}")
        else:
            print(f"ok: run[{adapter.name}] {len(page_names)} pages -> {out_dir}")
        return run
