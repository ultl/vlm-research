"""Qwen3-VL FP8 adapter using vLLM.

The official FP8 checkpoint is not loadable through the normal Transformers
model path yet, so this adapter follows Qwen's vLLM inference route while still
emitting the bakeoff Runner contract.
"""
from __future__ import annotations

import os
from typing import Dict

from .base import Adapter, InferResult


class Qwen3VLFP8VLLMAdapter(Adapter):
    name = "qwen3vl_fp8"
    accepts = "images"
    uses_prompt = True
    exposes_tokens = False

    def __init__(self, model_id: str, decode: dict):
        self.model_id = model_id
        self.decode = decode
        self.processor = None
        self.llm = None
        self.sampling_params = None
        self._cuda = None
        self._gpu_count = None

    def load(self) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self._cuda = torch.version.cuda
        self._gpu_count = torch.cuda.device_count()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3-VL FP8 requires a CUDA-visible vLLM environment, but "
                "torch.cuda.is_available() is False."
            )

        print(
            f"[{self.name}] loading processor for {self.model_id}...",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        tp_size = int(self.decode.get("tensor_parallel_size") or self._gpu_count or 1)
        gpu_memory_utilization = float(self.decode.get("gpu_memory_utilization", 0.70))
        enforce_eager = bool(self.decode.get("enforce_eager", False))
        print(
            f"[{self.name}] loading vLLM model tp={tp_size} "
            f"gpu_memory_utilization={gpu_memory_utilization}...",
            flush=True,
        )
        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tp_size,
            seed=int(self.decode.get("seed", 0)),
        )
        self.sampling_params = SamplingParams(
            temperature=float(self.decode.get("temperature", 0.0)),
            max_tokens=int(self.decode.get("max_new_tokens", 2048)),
            top_k=int(self.decode.get("top_k", -1)),
            stop_token_ids=self.decode.get("stop_token_ids", []),
        )

    def infer(self, image_path: str, prompt: str) -> InferResult:
        request = self._prepare_input(image_path, prompt)
        output = self.llm.generate([request], sampling_params=self.sampling_params)[0]
        return InferResult(text=output.outputs[0].text)

    def _prepare_input(self, image_path: str, prompt: str) -> dict:
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

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            video_kwargs = {}

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def env_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "model_id": self.model_id,
            "runtime": "vllm",
            "cuda": self._cuda,
            "gpu_count": self._gpu_count,
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
            import transformers
            info["transformers"] = transformers.__version__
            import vllm
            info["vllm"] = vllm.__version__
        except Exception:
            pass
        return info


def build(model_cfg: dict) -> Qwen3VLFP8VLLMAdapter:
    return Qwen3VLFP8VLLMAdapter(
        model_cfg["model_id"],
        model_cfg.get("decode", {}),
    )
