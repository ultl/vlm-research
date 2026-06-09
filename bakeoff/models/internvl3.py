"""InternVL3-2B adapter (HF transformers, end-to-end VLM, images path).

InternVL uses dynamic 448px tiling + a custom `model.chat()` API. The tiling
helpers below are the canonical InternVL preprocessing (from the model card);
only PIL is used there, so they're CPU-testable. Heavy deps (torch/torchvision)
import lazily so the module loads on a CPU box.
"""
from __future__ import annotations

from typing import Dict, List

from PIL import Image

from .base import Adapter, InferResult

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _closest_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12,
                       image_size=448, use_thumbnail=False) -> List[Image.Image]:
    w, h = image.size
    aspect = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    tar = _closest_ratio(aspect, ratios, w, h, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = []
    for i in range(blocks):
        box = ((i % cols) * image_size, (i // cols) * image_size,
               ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(image_file: str, input_size=448, max_num=12):
    import torch
    image = Image.open(image_file).convert("RGB")
    transform = _build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size,
                               use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


class InternVL3Adapter(Adapter):
    name = "internvl3"
    accepts = "images"
    uses_prompt = True
    exposes_tokens = True

    def __init__(self, model_id: str, decode: dict):
        self.model_id = model_id
        self.decode = decode
        self.model = None
        self.tokenizer = None
        self.device = None
        self._cuda = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._cuda = torch.version.cuda
        self.model = AutoModel.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.device = next(self.model.parameters()).device
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False)

    def infer(self, image_path: str, prompt: str) -> InferResult:
        import torch

        max_tiles = int(self.decode.get("max_tiles", 12))
        pixel_values = load_image(image_path, max_num=max_tiles).to(
            torch.bfloat16).to(self.device)
        gen_cfg = {
            "max_new_tokens": int(self.decode.get("max_new_tokens", 4096)),
            "do_sample": float(self.decode.get("temperature", 0.0)) > 0.0,
        }
        question = "<image>\n" + prompt
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, question, gen_cfg)

        out_tokens = None
        try:
            out_tokens = len(self.tokenizer(response).input_ids)
        except Exception:
            pass
        return InferResult(text=response, output_tokens=out_tokens)

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


def build(model_cfg: dict) -> InternVL3Adapter:
    return InternVL3Adapter(model_cfg["model_id"], model_cfg.get("decode", {}))
