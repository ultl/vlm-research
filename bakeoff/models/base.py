"""Adapter interface + the shared runner.

Adapters stay thin: declare capabilities, load() the model, and implement either
infer() (one page image) or infer_pdf() (whole PDF -> per-page). The Runner owns
everything uniform — timing, metadata, provenance, error capture, and writing the
output contract (pageN.md / pageN.meta.json / run.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    def infer(self, image_path: str, prompt: str) -> InferResult:
        """One page image (accepts='images')."""
        raise NotImplementedError(f"{self.name}: infer() not implemented")

    def infer_pdf(self, pdf_path: str, prompt: str) -> List[InferResult]:
        """Whole PDF -> one InferResult per page, in order (accepts='pdf')."""
        raise NotImplementedError(f"{self.name}: infer_pdf() not implemented")

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
        out_dir.mkdir(parents=True, exist_ok=True)
        started = _now()
        adapter.load()

        if adapter.accepts == "images":
            recs, errors, total = self._run_images(
                adapter, pages_dir, page_names, prompt, decode, out_dir)
        elif adapter.accepts == "pdf":
            recs, errors, total = self._run_pdf(
                adapter, pdf_path, pages_dir, page_names, prompt, decode, out_dir)
        else:
            raise ValueError(f"unknown accepts: {adapter.accepts!r}")

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
                       "total_latency_s": round(total, 3)},
            "pages": recs,
        }
        (out_dir / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

        if errors:
            print(f"run[{adapter.name}]: completed with {errors} error(s) -> {out_dir}")
        else:
            print(f"ok: run[{adapter.name}] {len(page_names)} pages -> {out_dir}")
        return run

    # ---- per-architecture drivers ----

    def _run_images(self, adapter, pages_dir, page_names, prompt, decode, out_dir):
        recs, errors, total = [], 0, 0.0
        for name in page_names:
            img = pages_dir / f"{name}.png"
            if not img.exists():
                raise SystemExit(f"run: page image missing: {img}")
            status, err = "ok", None
            res = InferResult(text="")
            t0 = time.perf_counter()
            try:
                res = adapter.infer(str(img), prompt)
            except Exception:
                status, err = "error", traceback.format_exc()
                errors += 1
            latency = time.perf_counter() - t0
            total += latency
            self._write_page(out_dir, name, img, res, latency, "page",
                             decode, status, err)
            recs.append({"page": name, "status": status,
                         "latency_s": round(latency, 3)})
        return recs, errors, total

    def _run_pdf(self, adapter, pdf_path, pages_dir, page_names, prompt, decode, out_dir):
        # Pipeline tools ingest the whole PDF once; latency is doc-level.
        results: List[InferResult] = []
        run_err = None
        t0 = time.perf_counter()
        try:
            results = adapter.infer_pdf(str(pdf_path), prompt)
        except Exception:
            run_err = traceback.format_exc()
        total = time.perf_counter() - t0
        per_page = total / len(page_names) if page_names else total

        recs, errors = [], 0
        for i, name in enumerate(page_names):
            img = pages_dir / f"{name}.png"
            if run_err is not None:
                status, err, res = "error", run_err, InferResult(text="")
            elif i >= len(results):
                status, err, res = "error", (
                    f"adapter returned {len(results)} pages, "
                    f"expected {len(page_names)}"), InferResult(text="")
            else:
                status, err, res = "ok", None, results[i]
            if status == "error":
                errors += 1
            self._write_page(out_dir, name, img if img.exists() else None,
                             res, per_page, "doc", decode, status, err)
            recs.append({"page": name, "status": status,
                         "latency_s": round(per_page, 3)})
        return recs, errors, total

    def _write_page(self, out_dir: Path, name: str, img: Optional[Path],
                    res: InferResult, latency: float, scope: str,
                    decode: dict, status: str, err: Optional[str]) -> None:
        (out_dir / f"{name}.md").write_text(res.text, encoding="utf-8")
        wh = list(Image.open(img).size) if img and img.exists() else None
        tps = (res.output_tokens / latency
               if res.output_tokens and latency > 0 else None)
        meta = {
            "page": name,
            "input_image": str(img.relative_to(self.root)) if img else None,
            "input_image_sha256": sha256_file(img) if img and img.exists() else None,
            "image_wh": wh,
            "latency_s": round(latency, 3),
            "latency_scope": scope,            # "page" (per-image) | "doc" (split)
            "ttft_s": res.ttft_s,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "tokens_per_s": round(tps, 2) if tps else None,
            "output_chars": len(res.text),
            "decode": decode,
            "status": status,
            "error": err,
        }
        (out_dir / f"{name}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
