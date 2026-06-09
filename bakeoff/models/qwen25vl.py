"""Qwen2.5-VL adapter (HF transformers).

Heavy deps (torch/transformers/qwen_vl_utils) import lazily inside load()/infer()
so this module stays importable on a CPU box for harness tests.
"""
from __future__ import annotations

import time
from typing import Dict

from .base import Adapter, InferResult


class Qwen25VLAdapter(Adapter):
    name = "qwen25vl"
    accepts = "images"
    uses_prompt = True
    exposes_tokens = True

    def __init__(self, model_id: str, decode: dict):
        self.model_id = model_id
        self.decode = decode
        self.model = None
        self.processor = None
        self._cuda = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._cuda = torch.version.cuda
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto")
        self.model.eval()
        # min/max pixels left at processor defaults; a dense scan benefits from
        # high resolution — revisit per-model if needed.
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def infer(self, image_path: str, prompt: str) -> InferResult:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": int(self.decode.get("max_new_tokens", 4096)),
            "do_sample": float(self.decode.get("temperature", 0.0)) > 0.0,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(self.decode["temperature"])

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        return InferResult(
            text=decoded,
            input_tokens=int(inputs.input_ids.shape[1]),
            output_tokens=int(trimmed.shape[1]),
        )

    def env_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {"model_id": self.model_id, "cuda": self._cuda}
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
            import transformers
            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info


def build(model_cfg: dict) -> Qwen25VLAdapter:
    return Qwen25VLAdapter(model_cfg["model_id"], model_cfg.get("decode", {}))
