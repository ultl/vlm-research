"""Qwen3-VL adapter (HF transformers) — the generational upgrade to qwen25vl.

Same images path + qwen_vl_utils flow + max_pixels cap as Qwen2.5-VL; only the
model class differs (and it needs a newer transformers). load() tries the
specific Qwen3-VL class, falling back to the generic image-text-to-text class.
Heavy deps import lazily so the module loads on CPU.
"""
from __future__ import annotations

from typing import Dict

from .base import Adapter, InferResult


class Qwen3VLAdapter(Adapter):
    name = "qwen3vl"
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
        from transformers import AutoProcessor

        self._cuda = torch.version.cuda
        try:                                  # VERIFY-ON-VM: class name for the pin
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except Exception:
            from transformers import AutoModelForImageTextToText as ModelCls
        self.model = ModelCls.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto")
        self.model.eval()
        kw = {}
        if self.decode.get("min_pixels"):
            kw["min_pixels"] = int(self.decode["min_pixels"])
        if self.decode.get("max_pixels"):
            kw["max_pixels"] = int(self.decode["max_pixels"])
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kw)

    def infer(self, image_path: str, prompt: str) -> InferResult:
        import torch
        from qwen_vl_utils import process_vision_info

        image_item = {"type": "image", "image": f"file://{image_path}"}
        if self.decode.get("min_pixels"):
            image_item["min_pixels"] = int(self.decode["min_pixels"])
        if self.decode.get("max_pixels"):
            image_item["max_pixels"] = int(self.decode["max_pixels"])
        messages = [{
            "role": "user",
            "content": [image_item, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": int(self.decode.get("max_new_tokens", 2048)),
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
            import transformers
            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info


def build(model_cfg: dict) -> Qwen3VLAdapter:
    return Qwen3VLAdapter(model_cfg["model_id"], model_cfg.get("decode", {}))
