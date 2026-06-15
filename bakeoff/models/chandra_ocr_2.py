"""Chandra OCR 2 adapter (Datalab document OCR, image path).

Chandra is a document-specialized OCR model that emits markdown/layout output.
By default this adapter uses Chandra's `prompt_type` presets. A model stanza can
also set `prompt_path`; in that case the runner-provided prompt is passed as a
custom Chandra prompt for Track A experiments.
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
                 prompt_type: str = "ocr_layout",
                 custom_prompt_enabled: bool = False,
                 max_output_tokens: int | None = None,
                 device_map: str = "cuda:0"):
        self.model_id = model_id
        self.method = method
        self.prompt_type = prompt_type
        self.custom_prompt_enabled = custom_prompt_enabled
        self.max_output_tokens = max_output_tokens
        self.device_map = device_map
        self.model = None
        self.manager = None
        self._cuda = None
        self._cuda_available = None

    def load(self) -> None:
        if self.method == "hf":
            self._load_hf()
        elif self.method == "vllm":
            self._load_vllm()
        else:
            raise ValueError(f"unknown Chandra method: {self.method!r}")

    def _load_hf(self) -> None:
        print(
            f"[{self.name}] importing torch/transformers for HF backend...",
            flush=True,
        )
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._cuda = torch.version.cuda
        self._cuda_available = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if self._cuda_available else "none"
        device_map = self._resolve_device_map(torch)
        print(
            f"[{self.name}] torch cuda_build={self._cuda} "
            f"cuda_available={self._cuda_available} gpu={gpu}; "
            f"loading weights for {self.model_id} with device_map={device_map}...",
            flush=True,
        )
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        loaded_map = getattr(self.model, "hf_device_map", None)
        if loaded_map:
            print(f"[{self.name}] loaded device map: {loaded_map}", flush=True)
        print(f"[{self.name}] model weights loaded; loading processor...", flush=True)
        self.model.eval()
        self.model.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.processor.tokenizer.padding_side = "left"
        print(f"[{self.name}] processor loaded", flush=True)

    def _resolve_device_map(self, torch):
        if self.device_map == "auto":
            return "auto"
        if self.device_map.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Chandra is configured for CUDA, but torch.cuda.is_available() "
                "is False in this process. Check the active env, CUDA build, "
                "and CUDA_VISIBLE_DEVICES."
            )
        return {"": self.device_map}

    def _load_vllm(self) -> None:
        from chandra.model import InferenceManager

        self.manager = InferenceManager(method="vllm")

    def infer(self, image_path: str, prompt: str) -> InferResult:
        custom_prompt = self._custom_prompt(prompt)
        if self.method == "hf":
            text = self._infer_hf(image_path, custom_prompt)
        else:
            text = self._infer_vllm(image_path, custom_prompt)
        return InferResult(text=text)

    def _infer_hf(self, image_path: str, custom_prompt: str | None) -> str:
        print(f"[{self.name}] importing Chandra HF helpers...", flush=True)
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown
        from PIL import Image

        prompt_source = "custom prompt" if custom_prompt else f"prompt_type={self.prompt_type}"
        print(f"[{self.name}] preparing image with {prompt_source}...", flush=True)
        with Image.open(image_path) as im:
            batch = [
                BatchInputItem(
                    image=im.convert("RGB"),
                    prompt=custom_prompt,
                    prompt_type=None if custom_prompt else self.prompt_type,
                )
            ]
            print(
                f"[{self.name}] generating page output "
                f"(max_output_tokens={self.max_output_tokens or 'package default'})...",
                flush=True,
            )
            result = generate_hf(
                batch,
                self.model,
                max_output_tokens=self.max_output_tokens,
            )[0]

        raw = getattr(result, "raw", None)
        if raw is not None:
            if custom_prompt:
                return self._as_text(raw)
            return self._as_text(parse_markdown(raw))
        markdown = getattr(result, "markdown", None)
        return self._as_text(markdown if markdown is not None else result)

    def _infer_vllm(self, image_path: str, custom_prompt: str | None) -> str:
        from chandra.model.schema import BatchInputItem
        from PIL import Image

        with Image.open(image_path) as im:
            batch = [
                BatchInputItem(
                    image=im.convert("RGB"),
                    prompt=custom_prompt,
                    prompt_type=None if custom_prompt else self.prompt_type,
                )
            ]
            result = self.manager.generate(
                batch,
                max_output_tokens=self.max_output_tokens,
            )[0]

        raw = getattr(result, "raw", None)
        if custom_prompt and raw is not None:
            return self._as_text(raw)
        markdown = getattr(result, "markdown", None)
        if markdown is not None:
            return self._as_text(markdown)
        return self._as_text(raw if raw is not None else result)

    def _custom_prompt(self, prompt: str) -> str | None:
        if not self.custom_prompt_enabled:
            return None
        prompt = prompt.strip()
        return prompt or None

    @staticmethod
    def _as_text(value) -> str:
        return value if isinstance(value, str) else str(value or "")

    def env_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "model_id": self.model_id,
            "method": self.method,
            "prompt_type": self.prompt_type,
            "custom_prompt_enabled": self.custom_prompt_enabled,
            "max_output_tokens": self.max_output_tokens,
            "device_map": self.device_map,
            "cuda": self._cuda,
            "cuda_available": self._cuda_available,
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
    decode = model_cfg.get("decode", {})
    max_output_tokens = decode.get("max_output_tokens", decode.get("max_new_tokens"))
    return ChandraOCR2Adapter(
        model_cfg["model_id"],
        method=model_cfg.get("method", "hf"),
        prompt_type=model_cfg.get("prompt_type", "ocr_layout"),
        custom_prompt_enabled=bool(
            model_cfg.get("custom_prompt") or model_cfg.get("prompt_path")
        ),
        max_output_tokens=max_output_tokens,
        device_map=model_cfg.get("device_map", "cuda:0"),
    )
