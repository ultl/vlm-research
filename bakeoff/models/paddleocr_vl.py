"""PaddleOCR-VL adapter (PaddlePaddle ecosystem, pipeline parser, pdf path).

PaddleOCR-VL is a 0.9B doc-parsing VLM in the Paddle stack — its own runtime, not
transformers, hence its own image. The pipeline entrypoint and per-page markdown
attribute are flagged VERIFY-ON-VM (Paddle's API moves between releases).
"""
from __future__ import annotations

from typing import Dict, List

from .base import Adapter, InferResult


class PaddleOCRVLAdapter(Adapter):
    name = "paddleocr_vl"
    accepts = "pdf"
    uses_prompt = False
    exposes_tokens = False

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipeline = None

    def load(self) -> None:
        # VERIFY-ON-VM: pipeline entrypoint for the installed PaddleOCR-VL release.
        from paddleocr import PaddleOCRVL
        self.pipeline = PaddleOCRVL()

    def infer_pdf(self, pdf_path: str, prompt: str) -> List[InferResult]:
        results = self.pipeline.predict(pdf_path)
        return [InferResult(text=self._page_markdown(res)) for res in results]

    @staticmethod
    def _page_markdown(res) -> str:
        # VERIFY-ON-VM: markdown attribute shape across PaddleOCR-VL versions.
        md = getattr(res, "markdown", None)
        if isinstance(md, dict):
            return md.get("markdown_texts") or md.get("text") or ""
        if isinstance(md, str):
            return md
        j = getattr(res, "json", None)
        if isinstance(j, dict):
            return j.get("markdown") or ""
        return str(md) if md else ""

    def env_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {"model_id": self.model_id}
        try:
            import paddle
            info["paddle"] = paddle.__version__
            info["cuda_compiled"] = paddle.device.is_compiled_with_cuda()
        except Exception:
            pass
        return info


def build(model_cfg: dict) -> PaddleOCRVLAdapter:
    return PaddleOCRVLAdapter(model_cfg["model_id"])
