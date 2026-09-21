# R²-SFT Data Pipeline

This directory reproduces the data side of the R²-SFT pipeline (paper §4):
target-blind CoT synthesis → PPL scorer warm-up → SID-PPL CoT filtering → target-aware SFT data.

## Layout

```
cot/
├── api.py                       # OpenAI-compatible LLM client (for step 2)
├── 0_caption_summary/          # Step 0: summarize pid2caption with Qwen3-8B
├── 1_history_text/              # Step 1: extract per-user history+caption text
├── 2_cot/                       # Step 2: teacher LLM target-blind CoT synthesis
├── 3_ppl_eval/                  # Step 3: PPL preprocess + eval + join
│   └── main_version/            #   main recipe (steps 0-5 wrappers)
├── 4_filter_ppl/                # Step 4: per-target PPL-gain CoT filter
└── 5_data_format/               # Step 5: final SFT dataset assembly
```

All paths below are **relative to this `cot/` directory**.

## Prerequisites

Raw data (under `../../RecIF/`):
- `onerec_bench_release.parquet` — user history + targets
- `video_ad_pid2sid.parquet` — video/ad pid → SID map
- `pid2caption.parquet` — video/ad pid → caption (input to step 0)

Base model: `OpenOneRec-8B-Pretrain` (HuggingFace) → converted to local `models/pretrain-8B/converted/`.

MPI hostfile at `/etc/mpi/hostfile` (override with `$HOSTFILE`).

## Pipeline (data side)

### Step 0 — caption summary (optional, saves API cost in step 2)

Summarize `pid2caption.parquet` into short `summary` captions with a local Qwen3-8B.

```bash
bash 0_caption_summary/run_mpi.sh
```

Output: `../../RecIF-cot/caption_summary/`

> Skip this if you want to feed full `dense_caption` to the teacher in step 2.

### Step 1 — history + caption text extraction

Per-user extraction of history pids + captions (video + ad), fed to the teacher LLM in step 2.

```bash
bash 1_history_text/run_mpi.sh
```

Output: `../../RecIF-cot/steps/1_history_text/`

### Step 2 — target-blind CoT synthesis

Teacher LLM synthesizes a rationale from the history captions **without seeing the target**.
Fill in `2_cot/api_config.json` with your OpenAI-compatible endpoint first:

```json
{ "base_url": "https://api.openai.com/v1", "api_key": "your_token", "model": "gpt-4o-mini" }
```

Then:

```bash
API_CONFIG_PATH=2_cot/api_config.json bash 2_cot/run.sh
```

Output: `../../RecIF-cot/steps/2_cot/` (video + ad CoT)

### Step 0' — build warm-up scorer SFT data (raw, unfiltered CoT)

Build the SFT training data for the 200-step warm-up scorer (used as the PPL scorer in step 3).
Uses the raw CoT from step 2 (no PPL filtering yet).

```bash
bash 3_ppl_eval/main_version/0_build_warmup_data.sh
```

Output: `../../output/r2sft_warmup_raw/` + blended json `r2sft_warmup_raw.json`

### Train the warm-up scorer (200 steps) — training side

```bash
cd ../../../pretrain
python3 auto_train/auto_train.py auto_train/r2sft_warmup_200steps.yaml
```

This produces `models/r2sft_warmup_200steps/step200/converted/`,
which step 3 uses as the PPL scorer.

### Step 3 — PPL preprocess + eval + join

Score the target-SID PPL for each candidate CoT with the warm-up scorer:

```bash
bash 3_ppl_eval/main_version/1_preprocess.sh   # explode CoT into per-target PPL inputs
bash 3_ppl_eval/main_version/2_eval_ppl.sh      # score with step200 scorer (multi-GPU)
bash 3_ppl_eval/main_version/3_join.sh         # join back with step2 CoT (per-target)
```

Outputs (under `../../RecIF-cot/steps/3_ppl_eval/`):
- `preprocessed_sid_cot_ver5/`
- `ppl_eval_sft_cot_ver5/`
- `joined_sft_ver5/`

### Step 4 — per-target PPL filter

Two-phase: first annotate to get the `log_ppl_cot` distribution, then threshold at p50.

```bash
# 1) annotate all (no threshold) to get distribution
bash 3_ppl_eval/main_version/4_filter.sh
# → read log_ppl_cot_stats.p50 from:
#   ../../RecIF-cot/steps/4_filter_ppl/r2sft_filter/filter_summary.json

# 2) threshold filter at p50
MAX_LOG_PPL_COT=<p50> bash 3_ppl_eval/main_version/4_filter.sh
```

Output: `../../RecIF-cot/steps/4_filter_ppl/r2sft_filter/`

### Step 5 — final SFT dataset (★ main output)

Assemble the filtered CoT + non-CoT rec + caption2sid/sid2caption into the final
sharded SFT dataset with `file_list.json`:

```bash
bash 3_ppl_eval/main_version/5_data_format.sh
```

Output: `../../output/r2sft_sft_ppl/` + blended json `r2sft_sft_ppl.json`

> Or use the orchestrator for steps 3→5 in one go (after the warm-up scorer is trained):
> ```bash
> MAX_LOG_PPL_COT=<p50> bash 3_ppl_eval/main_version/run_all.sh
> ```

## Train the main SFT (r2sft_sft_ppl) — training side

```bash
cd ../../../pretrain
python3 auto_train/auto_train.py auto_train/r2sft_sft_ppl.yaml
```

Uses `examples/dataset_config/r2sft_sft_ppl.json` →
`examples/blended/r2sft_sft_ppl.json` (from step 5).

Key training args (target-aware SFT, paper §4.4):
- `ENABLE_TARGET_LOSS=1`, `TARGET_LOSS_SOURCES_NAME="RecIF_RecCOT"`, `TARGET_LOSS_BETA=1.0`
- `ENABLE_THINK_LOSS=0`
- `num_training_steps=6600`, `save_checkpoint_per_step=1650`, `max_step=8250`

## Outputs summary

| stage | path (relative to `cot/`) |
|-------|----------------------------|
| step 0 caption summary | `../../RecIF-cot/caption_summary/` |
| step 1 history text | `../../RecIF-cot/steps/1_history_text/` |
| step 2 CoT | `../../RecIF-cot/steps/2_cot/` |
| warm-up data | `../../output/r2sft_warmup_raw/` |
| step 3 preprocess | `../../RecIF-cot/steps/3_ppl_eval/preprocessed_sid_cot_ver5/` |
| step 3 PPL eval | `../../RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_cot_ver5/` |
| step 3 join | `../../RecIF-cot/steps/3_ppl_eval/joined_sft_ver5/` |
| step 4 filter | `../../RecIF-cot/steps/4_filter_ppl/r2sft_filter/` |
| step 5 final SFT | `../../output/r2sft_sft_ppl/` |
| blended json (warm-up) | `../../../pretrain/examples/blended/r2sft_warmup_raw.json` |
| blended json (main SFT) | `../../../pretrain/examples/blended/r2sft_sft_ppl.json` |
