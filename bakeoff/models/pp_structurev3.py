"""PP-StructureV3 adapter — classic modular document-parsing pipeline (pdf path).

The pre-VLM baseline: orientation/dewarp + PP-OCRv5 + layout + table/formula/chart
/seal models assembled into a pipeline — no VLM at the recognition core. Same
Paddle stack as paddleocr_vl (same package/env). VERIFY-ON-VM: PPStructureV3
entrypoint + per-page markdown attribute.
"""
from __future__ import annotations

from typing import Dict, List

from .base import Adapter, InferResult


class PPStructureV3Adapter(Adapter):
    name = "pp_structurev3"
    accepts = "pdf"
    uses_prompt = False          # pure modular pipeline — no prompt at all
    exposes_tokens = False

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipeline = None

    def load(self) -> None:
        from paddleocr import PPStructureV3
        self.pipeline = PPStructureV3()

    def infer_pdf(self, pdf_path: str, prompt: str) -> List[InferResult]:
        results = self.pipeline.predict(pdf_path)
        return [InferResult(text=self._page_markdown(res)) for res in results]

    @staticmethod
    def _page_markdown(res) -> str:
        # VERIFY-ON-VM: markdown attribute shape across paddleocr versions.
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


def build(model_cfg: dict) -> PPStructureV3Adapter:
    return PPStructureV3Adapter(model_cfg["model_id"])
