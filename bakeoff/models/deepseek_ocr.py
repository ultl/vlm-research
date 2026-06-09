"""DeepSeek-OCR adapter (HF transformers, optical-compression OCR, images path).

Uses DeepSeek-OCR's custom `model.infer()`. Its stack is pinned differently from
the other VLMs (transformers 4.46.x, torch 2.6) — hence its own Docker image. The
infer() call signature + result handling are flagged VERIFY-ON-VM.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

from .base import Adapter, InferResult


class DeepSeekOCRAdapter(Adapter):
    name = "deepseek_ocr"
    accepts = "images"
    uses_prompt = True
    exposes_tokens = True

    def __init__(self, model_id: str, decode: dict):
        self.model_id = model_id
        self.decode = decode
        self.model = None
        self.tokenizer = None
        self._cuda = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._cuda = torch.version.cuda
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, use_safetensors=True,
            _attn_implementation=self.decode.get("attn", "eager")).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.bfloat16)

    def infer(self, image_path: str, prompt: str) -> InferResult:
        # DeepSeek-OCR grounding-markdown template; inject the shared instruction.
        full = f"<image>\n<|grounding|>{prompt.strip()}"
        out = tempfile.mkdtemp(prefix="dsocr_")
        # VERIFY-ON-VM: model.infer signature/return for the pinned release.
        res = self.model.infer(
            self.tokenizer, prompt=full, image_file=image_path, output_path=out,
            base_size=int(self.decode.get("base_size", 1024)),
            image_size=int(self.decode.get("image_size", 640)),
            crop_mode=bool(self.decode.get("crop_mode", True)),
            save_results=True, test_compress=False)
        text = res if isinstance(res, str) and res.strip() else self._read_result(out)

        out_tokens = None
        try:
            out_tokens = len(self.tokenizer(text).input_ids)
        except Exception:
            pass
        return InferResult(text=text, output_tokens=out_tokens)

    @staticmethod
    def _read_result(out_dir: str) -> str:
        import glob
        import os
        for pat in ("*.mmd", "*.md", "result*.txt"):
            hits = sorted(glob.glob(os.path.join(out_dir, "**", pat), recursive=True))
            if hits:
                return Path(hits[0]).read_text(encoding="utf-8")
        return ""

    def env_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {"model_id": self.model_id, "cuda": self._cuda}
        try:
            import torch
            import transformers
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info


def build(model_cfg: dict) -> DeepSeekOCRAdapter:
    return DeepSeekOCRAdapter(model_cfg["model_id"], model_cfg.get("decode", {}))
