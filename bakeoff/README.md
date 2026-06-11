# VLM OCR bake-off

Run several open-source document VLMs on the scanned `sample.pdf` and score them
against the hand-verified gold in `fixtures/gt/`. Develop on CPU, run on a GPU VM.

## Quickstart — native on the GPU VM (no Docker, recommended)
Light path: reuse your env's torch, no image builds. Copy-paste, top to bottom:
```sh
# 0. one-time env — Python 3.11 (NOT 3.13/3.14: torch has no wheels there yet; 3.10 also fine)
micromamba create -n vlm python=3.11 -y && micromamba activate vlm
pip install torch torchvision transformers==4.51.3 accelerate timm einops \
            sentencepiece qwen-vl-utils pillow      # plain torch — NO --index-url on a working VM

# 1. weights cache on a big disk (default ~/.cache is usually too small)
export HF_HOME=/mnt/data/<you>/hf_cache             # add to ~/.bashrc to persist

# 2. data — fixtures/ is already in the repo; only the raw PDF needs copying over
ls sample.pdf || echo "scp sample.pdf to the repo root first"

# 3. run + score; pick a free GPU (nvidia-smi to check which)
CUDA_VISIBLE_DEVICES=0 bakeoff/run_native.sh internvl3 qwen25vl
python bakeoff/report.py
```
That covers the two transformers VLMs. The other three need their own env (see
"Running without Docker" below) or Docker. See "Gotchas" if anything errors.

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

## Running without Docker (micromamba)
The harness runs natively — Docker only *supplies* isolated deps. If your env has
a model's deps, run it directly (reuses your torch, no image build, no multi-GB
download):
```sh
micromamba activate <env>
bakeoff/run_native.sh                 # internvl3 + qwen25vl (shared transformers stack)
bakeoff/run_native.sh internvl3       # just one
```
The five models split into **conflicting stacks**, so one env can't hold all.
Make a per-model env for the others (lighter than Docker, you manage the envs):
```sh
# transformers VLMs — internvl3, qwen25vl
micromamba create -n vlm python=3.11 -c conda-forge && micromamba activate vlm
pip install torch torchvision transformers==4.51.3 accelerate timm einops \
            sentencepiece qwen-vl-utils pillow
bakeoff/run_native.sh internvl3 qwen25vl

# DeepSeek-OCR — older transformers + torch 2.6
micromamba create -n dsocr python=3.11 -c conda-forge && micromamba activate dsocr
pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict pillow
bakeoff/run_native.sh deepseek_ocr

# PaddleOCR-VL — Paddle stack (cu118 wheel; no cu121 build exists)
micromamba create -n paddle python=3.11 -c conda-forge && micromamba activate paddle
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install -U paddleocr        # needs the PaddleOCRVL class (>=3.6, ships VL-1.6)
python -c "from paddleocr import PaddleOCRVL"   # sanity-check the import
bakeoff/run_native.sh paddleocr_vl

# MinerU — its own pinned stack
micromamba create -n mineru python=3.11 -c conda-forge && micromamba activate mineru
pip install "mineru[core]==2.5.4"
bakeoff/run_native.sh mineru
```
Scoring is env-agnostic, so `report.py` merges results across all of them at the end.

## Reproducibility
- Pin the base image by digest for a frozen run; record CUDA/GPU (captured in `run.json` via the adapter's `env_info`).
- Decode is greedy (`temperature=0`) and recorded per page in `meta.json`.

## Gotchas (hit once, fixed — keep for next time)
| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for torch==2.5.1` | Python 3.13/3.14 (no wheels) or wrong arch | use Python **3.11**; native install is just `pip install torch torchvision` (no `--index-url`) |
| `apt-get … connection timed out` in a Docker build | VM blocks outbound port 80 | already fixed — Dockerfiles use **https** apt mirrors |
| `paddlepaddle-gpu==3.0.0` not found | Paddle ships no `cu121` wheels | already fixed — Dockerfile uses the **cu118** index |
| weights fill the disk / very slow first run | HF cache on a small root disk | `export HF_HOME=/big/disk/hf_cache` **before** the first run |
| both GPUs used, or the wrong one | default device pick (`device_map="auto"` spans all) | prefix the command with `CUDA_VISIBLE_DEVICES=N` |
| `score: no run found for '<model>'` | scoring where no `runs/<model>/` exists | run first (GPU box), then score there, or `scp -r …/runs/<model> ./runs/` |
| run errored but others still ran | per-page error capture (by design) | `python bakeoff/score.py` skips it; read `runs/<model>/page1.meta.json` → `.error` |

## Adding a model
Add a `models/<name>.py` exposing `build(model_cfg) -> Adapter`, a stanza in
`config.json`, a `docker/Dockerfile.<name>`, and a `docker-compose.yml` service.
Pipeline tools (`accepts: "pdf"`) need the pdf-path branch in the runner.
