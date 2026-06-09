# VLM OCR bake-off

Run several open-source document VLMs on the scanned `sample.pdf` and score them
against the hand-verified gold in `fixtures/gt/`. Develop on CPU, run on a GPU VM.

## Models (5)
| Key | Model | Size | Path | Notes |
|---|---|---|---|---|
| `qwen25vl` | Qwen2.5-VL-7B | 7B | images | VRAM ceiling (~16–20 GB) |
| `internvl3` | InternVL3-2B | 2B | images | dynamic tiling (`max_tiles`) |
| `deepseek_ocr` | DeepSeek-OCR | ~3B | images | optical compression; transformers 4.46 pin |
| `mineru` | MinerU2.5 | 1.2B | pdf | pipeline parser; `*VERIFY-ON-VM*` CLI |
| `paddleocr_vl` | PaddleOCR-VL | 0.9B | pdf | Paddle stack; `*VERIFY-ON-VM*` pipeline |

Each runs in its own Docker image (incompatible dep stacks). `VERIFY-ON-VM` marks
model-specific calls (CLI flags / API) whose exact form may need a tweak on first
run; the harness integration around them is validated.

## Pipeline
```
render.py   pdf -> fixtures/pages/*.png          (CPU)
run.py      pages -> runs/<model>/ (the contract) (GPU, in Docker)
score.py    runs/ vs fixtures/gt/cells -> scores/ (CPU)
report.py   scores/ -> scorecard                  (CPU)
```

## Output contract (`runs/<model>/`)
- `pageN.md` — model output verbatim (normalization happens only in `score.py`)
- `pageN.meta.json` — latency, tokens, image hash, decode params, status
- `run.json` — provenance: model + image digest + code SHA + prompt + pdf hash

## Data (PII — repo is PRIVATE)
This repo holds a real tax return (PII); keep it **private**. `fixtures/` (gold +
page images) is tracked here. The raw `sample.pdf` is gitignored — `render.py`
regenerates `fixtures/pages/` from it, so supply it out-of-band if you need to
re-render:
```sh
scp sample.pdf user@vm:/path/vlm-research/
```
The gold (`fixtures/gt/`) is already in the repo, so most runs need no scp.

## CPU side (no GPU needed)
```sh
python3 bakeoff/render.py                 # regenerate page PNGs
python3 bakeoff/score.py --model qwen25vl  # after a run exists
python3 bakeoff/report.py
```

## GPU VM
Requires Docker + the NVIDIA Container Toolkit (`nvidia-ctk`). Single 24 GB GPU
(L4 / A10 / 4090) covers the 7B at fp16.

One command does everything (build → run → score every model → scorecard):
```sh
bakeoff/run_all.sh                       # all five, smallest-VRAM first
bakeoff/run_all.sh paddleocr_vl mineru internvl3 deepseek_ocr   # skip the 7B on a small GPU
```
It runs models one at a time (each gets the full GPU) and skips past any that fail.

Or step through one model manually:
```sh
export BAKEOFF_CODE_SHA=$(git rev-parse HEAD)
docker compose -f bakeoff/docker-compose.yml build qwen25vl
docker compose -f bakeoff/docker-compose.yml run --rm qwen25vl   # writes runs/qwen25vl/
```
Weights download once into the `hf_cache` volume and persist across runs.

Then score (VM or pull `runs/` back to CPU):
```sh
python3 bakeoff/score.py --model qwen25vl
python3 bakeoff/report.py
```

## Reproducibility
- Pin the base image by digest for a frozen run; record CUDA/GPU (captured in `run.json` via the adapter's `env_info`).
- Decode is greedy (`temperature=0`) and recorded per page in `meta.json`.

## Adding a model
Add a `models/<name>.py` exposing `build(model_cfg) -> Adapter`, a stanza in
`config.json`, a `docker/Dockerfile.<name>`, and a `docker-compose.yml` service.
Pipeline tools (`accepts: "pdf"`) need the pdf-path branch in the runner.
