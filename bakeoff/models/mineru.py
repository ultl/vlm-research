"""MinerU 2.5 adapter — pipeline document parser (accepts a whole PDF).

MinerU ships a CLI whose output layout is more stable than its Python API across
versions, so we shell out and read the result. The two MinerU-specific bits —
the CLI invocation and the output file layout — are isolated below and flagged
VERIFY-ON-VM, because MinerU's flags/paths have changed between releases.

Per-page split: MinerU emits a `*_content_list.json` of blocks each carrying a
`page_idx`; we group blocks by page and rebuild markdown per page. Falls back to
the whole-doc `.md` as a single page if the content list isn't found.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from .base import Adapter, InferResult


class MinerUAdapter(Adapter):
    name = "mineru"
    accepts = "pdf"
    uses_prompt = False          # fixed-behavior parser, no free-text prompt
    exposes_tokens = False

    def __init__(self, model_id: str, backend: str = "vlm-transformers"):
        self.model_id = model_id
        self.backend = backend

    def load(self) -> None:
        # MinerU downloads its weights on first CLI run (into HF_HOME). No-op here.
        pass

    def infer_pdf(self, pdf_path: str, prompt: str) -> List[InferResult]:
        out = tempfile.mkdtemp(prefix="mineru_")
        # VERIFY-ON-VM: flags for MinerU 2.5. Expected form:
        #   mineru -p <pdf> -o <out> -b vlm-transformers
        cmd = ["mineru", "-p", pdf_path, "-o", out, "-b", self.backend]
        print(f"  $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)   # stream MinerU's own progress/download live

        content_list = self._find(out, "*_content_list.json")
        if content_list:
            return self._split_by_page(content_list)
        # fallback: single whole-doc markdown as one page
        md = self._find(out, "*.md")
        text = Path(md).read_text(encoding="utf-8") if md else ""
        return [InferResult(text=text)]

    # ---- MinerU output handling (VERIFY-ON-VM) ----

    @staticmethod
    def _find(root: str, pattern: str):
        hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        return hits[0] if hits else None

    @staticmethod
    def _block_to_md(block: dict) -> str:
        t = block.get("type")
        if t == "text":
            return block.get("text", "")
        if t == "table":
            return block.get("table_body", "") or block.get("text", "")
        if t in ("title", "header"):
            return "# " + block.get("text", "")
        if t == "equation":
            return block.get("text", "")
        return block.get("text", "") or block.get("table_body", "") or ""

    def _split_by_page(self, content_list_path: str) -> List[InferResult]:
        blocks = json.loads(Path(content_list_path).read_text(encoding="utf-8"))
        pages: Dict[int, List[str]] = {}
        for b in blocks:
            idx = b.get("page_idx", 0)
            pages.setdefault(idx, []).append(self._block_to_md(b))
        return [InferResult(text="\n\n".join(pages[i]).strip())
                for i in sorted(pages)]

    def env_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {"model_id": self.model_id, "backend": self.backend}
        try:
            v = subprocess.run(["mineru", "--version"], capture_output=True, text=True)
            info["mineru_version"] = v.stdout.strip() or v.stderr.strip()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["cuda"] = torch.version.cuda
        except Exception:
            pass
        return info


def build(model_cfg: dict) -> MinerUAdapter:
    return MinerUAdapter(model_cfg["model_id"],
                         model_cfg.get("backend", "vlm-transformers"))
