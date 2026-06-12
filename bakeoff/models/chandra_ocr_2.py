"""Chandra OCR 2 adapter (Datalab document OCR, image path).

Chandra is a document-specialized OCR model that emits markdown/layout output.
Its public package exposes a higher-level `BatchInputItem` API rather than a
plain transformers chat prompt, so this adapter ignores the shared free-text
prompt and uses Chandra's `ocr_layout` prompt type.
"""
from __future__ import annotations

from typing import Dict

from .base import Adapter, InferResult


class ChandraOCR2Adapter(Adapter):
    name = "chandra_ocr_2"
    accepts = "images"
    uses_prompt = False
    exposes_tokens = False

    def __init__(self, model_id: str, method: str = "hf",
                 prompt_type: str = "ocr_layout"):
        self.model_id = model_id
        self.method = method
        self.prompt_type = prompt_type
        self.model = None
        self.manager = None
        self._cuda = None

    def load(self) -> None:
        if self.method == "hf":
            self._load_hf()
        elif self.method == "vllm":
            self._load_vllm()
        else:
            raise ValueError(f"unknown Chandra method: {self.method!r}")

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._cuda = torch.version.cuda
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.model.eval()
        self.model.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.processor.tokenizer.padding_side = "left"

    def _load_vllm(self) -> None:
        from chandra.model import InferenceManager

        self.manager = InferenceManager(method="vllm")

    def infer(self, image_path: str, prompt: str) -> InferResult:
        if self.method == "hf":
            text = self._infer_hf(image_path)
        else:
            text = self._infer_vllm(image_path)
        return InferResult(text=text)

    def _infer_hf(self, image_path: str) -> str:
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown
        from PIL import Image

        with Image.open(image_path) as im:
            batch = [
                BatchInputItem(
                    image=im.convert("RGB"),
                    prompt_type=self.prompt_type,
                )
            ]
            result = generate_hf(batch, self.model)[0]

        raw = getattr(result, "raw", None)
        if raw is not None:
            return self._as_text(parse_markdown(raw))
        markdown = getattr(result, "markdown", None)
        return self._as_text(markdown if markdown is not None else result)

    def _infer_vllm(self, image_path: str) -> str:
        from chandra.model.schema import BatchInputItem
        from PIL import Image

        with Image.open(image_path) as im:
            batch = [
                BatchInputItem(
                    image=im.convert("RGB"),
                    prompt_type=self.prompt_type,
                )
            ]
            result = self.manager.generate(batch)[0]

        markdown = getattr(result, "markdown", None)
        if markdown is not None:
            return self._as_text(markdown)
        raw = getattr(result, "raw", None)
        return self._as_text(raw if raw is not None else result)

    @staticmethod
    def _as_text(value) -> str:
        return value if isinstance(value, str) else str(value or "")

    def env_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "model_id": self.model_id,
            "method": self.method,
            "prompt_type": self.prompt_type,
            "cuda": self._cuda,
        }
        try:
            import importlib.metadata
            info["chandra_ocr"] = importlib.metadata.version("chandra-ocr")
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass
        try:
            import transformers
            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info


def build(model_cfg: dict) -> ChandraOCR2Adapter:
    return ChandraOCR2Adapter(
        model_cfg["model_id"],
        method=model_cfg.get("method", "hf"),
        prompt_type=model_cfg.get("prompt_type", "ocr_layout"),
    )
