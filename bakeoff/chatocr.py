#!/usr/bin/env python3
"""PP-ChatOCRv4Doc — RAG-KIE field extraction (parse -> retrieve -> LOCAL LLM).

Paddle's key-information-extraction pipeline: it parses the PDF (PP-Structure +
PP-DocBee2), builds a vector index, and asks an LLM to extract the requested key
fields. We point it at a LOCAL OpenAI-compatible LLM, so the document never
leaves the machine (PII-safe — no Baidu cloud).

PREREQ: a local OpenAI-compatible LLM server, e.g.
    vllm serve <model> --port 8000
then set (operator-local — never commit these):
    export CHATOCR_LLM_BASE_URL=http://127.0.0.1:8000/v1
    export CHATOCR_LLM_API_KEY=EMPTY
    export CHATOCR_LLM_MODEL=<served-model-name>
Optional separate embedding endpoint (defaults to the LLM one):
    export CHATOCR_EMB_BASE_URL / CHATOCR_EMB_API_KEY / CHATOCR_EMB_MODEL

VERIFY-ON-VM: the PPChatOCRv4Doc visual_predict / build_vector / chat API and
the result dict keys vary across paddleocr releases.

Usage: python bakeoff/chatocr.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import to_int, flatten_gold, bucket  # reuse normalization + bucketing
from kie import is_blank                         # blank/null detection

ROOT = Path(__file__).resolve().parent.parent
NAME = "pp_chatocrv4"


def env_chat_config(prefix: str, fallback: dict | None = None) -> dict:
    base = os.environ.get(f"{prefix}_BASE_URL")
    if not base:
        return dict(fallback) if fallback else {}
    return {
        "api_type": "openai",
        "base_url": base,
        "api_key": os.environ.get(f"{prefix}_API_KEY", "EMPTY"),
        "model_name": os.environ.get(f"{prefix}_MODEL", "default"),
    }


def gold_fields():
    """Scoreable, labelled cells across all pages (filled GOLD/CANDIDATE + empty)."""
    cfg = json.loads((ROOT / "bakeoff" / "config.json").read_text())
    gdir = ROOT / cfg["paths"]["gold"]
    out = []
    for name in cfg["pages"]:
        for c in flatten_gold(json.loads((gdir / f"{name}.cells.json").read_text())):
            if c.get("label") and bucket(c) in ("gold", "candidate", "gold_empty"):
                c["page"] = name
                out.append(c)
    return cfg, out


def score(chat_res: dict, fields) -> dict:
    """chat_res is {label: value}. Match each gold field by its label."""
    rows = []
    for c in fields:
        b, label = bucket(c), c["label"]
        present = label in chat_res
        raw = chat_res.get(label)
        if b in ("gold", "candidate"):
            gi = to_int(c["value"])
            pv = to_int(raw) if present else None
            verdict = ("missed" if pv is None else
                       "correct" if pv == gi else "wrong")
        else:
            verdict = "clean" if (not present or is_blank(raw)) else "hallucinated"
        rows.append({"page": c["page"], "id": c.get("id"), "label": label,
                     "bucket": b, "gold": c["value"],
                     "pred": raw if present else None, "verdict": verdict})
    return rows


def main() -> int:
    chat_bot = env_chat_config("CHATOCR_LLM")
    if not chat_bot:
        sys.exit("chatocr: set CHATOCR_LLM_BASE_URL (a local OpenAI-compatible "
                 "LLM endpoint) — see the module docstring")
    retriever = env_chat_config("CHATOCR_EMB", fallback=chat_bot)

    cfg, fields = gold_fields()
    key_list = sorted({c["label"] for c in fields})
    pdf = str(ROOT / cfg["paths"]["pdf"])
    print(f"[{NAME}] {len(key_list)} keys, LLM={chat_bot['base_url']}", flush=True)

    from paddleocr import PPChatOCRv4Doc  # VERIFY-ON-VM
    pipe = PPChatOCRv4Doc()

    print(f"[{NAME}] visual_predict (parse PDF)…", flush=True)
    t0 = time.perf_counter()
    visual_info = [r["visual_info"] for r in pipe.visual_predict(pdf)]
    print(f"[{NAME}] build_vector (index)…", flush=True)
    vector_info = pipe.build_vector(visual_info, retriever_config=retriever)
    print(f"[{NAME}] chat (extract {len(key_list)} keys via local LLM)…", flush=True)
    chat_out = pipe.chat(key_list=key_list, visual_info=visual_info,
                         vector_info=vector_info, chat_bot_config=chat_bot,
                         retriever_config=retriever)
    dt = time.perf_counter() - t0
    chat_res = chat_out.get("chat_res", chat_out) if isinstance(chat_out, dict) else {}

    out_dir = ROOT / cfg["paths"]["runs"] / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chat_res.json").write_text(
        json.dumps(chat_res, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = score(chat_res, fields)
    filled = [r for r in rows if r["bucket"] in ("gold", "candidate")]
    empt = [r for r in rows if r["bucket"] == "gold_empty"]

    def rate(rs, good):
        return round(sum(1 for r in rs if r["verdict"] in good) / len(rs), 3) if rs else None

    metrics = {
        "field_accuracy": rate(filled, ("correct",)),
        "field_n": len(filled),
        "field_correct": sum(1 for r in filled if r["verdict"] == "correct"),
        "empty_clean_rate": rate(empt, ("clean",)),
        "empty_n": len(empt),
        "hallucinated": sum(1 for r in empt if r["verdict"] == "hallucinated"),
        "latency_s": round(dt, 1),
    }
    result = {"model": NAME, "task": "rag-kie", "metrics": metrics, "fields": rows,
              "llm": chat_bot["base_url"]}
    sdir = ROOT / cfg["paths"]["scores"]
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"{NAME}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ok: {NAME} field_acc {metrics['field_accuracy']} "
          f"({metrics['field_correct']}/{metrics['field_n']}) | "
          f"empty_clean {metrics['empty_clean_rate']} in {dt:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
