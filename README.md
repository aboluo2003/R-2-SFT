<div align="center">
  <h1>Beyond Post-hoc Rationales: Improving Reasoning SFT for LLM-based Recommendation</h1>
  <p align="center"><a href="#license"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green" /></a></p>
</div>
<br>

## 📖 Introduction

**R²-SFT** is a target-aware SFT recipe for LLM-based recommendation that teaches
a recommendation LLM to produce faithful chain-of-thought (CoT) rationales before
its item predictions. The pipeline has four stages:

1. **Target-blind CoT synthesis** — a teacher LLM generates a rationale from the
   user's history *without seeing the target*, so the rationale cannot leak the
   answer.
2. **PPL scorer warm-up** — a 200-step SFT checkpoint is trained on the raw CoT
   data and reused as a per-target SID-PPL scorer.
3. **SID-PPL CoT filtering** — each candidate CoT is scored by
   `log P(target | history, CoT)` vs. `log P(target | history)`, and only CoT
   with positive PPL gain is kept (threshold at the p50 of the distribution).
4. **Target-aware SFT** — the main model is fine-tuned on the filtered CoT with a
   target-aware loss (`ENABLE_TARGET_LOSS=1`) that only computes the loss on the
   target tokens of CoT samples.

## 📂 Repository Layout

| Directory | Description | Docs |
|-----------|-------------|------|
| `data/onerec_data/sft/cot/` | **R²-SFT data pipeline** (steps 0–5): CoT synthesis → PPL scorer warm-up → PPL filter → final SFT dataset | [`README`](data/onerec_data/sft/cot/README.md) ★ |
| `pretrain/` | Training code (Qwen3-based, MPI + FSDP). Includes the R²-SFT warm-up scorer and main SFT recipes | [`README`](pretrain/README.md) |
| `benchmarks/` | RecIF-Bench evaluation harness (Ray + vLLM) | [`README`](benchmarks/README.md) |
| `data/` | Raw data format specs and general-text / pretraining data processing | [`README`](data/README.md) |
| `tokenizer/` | Tokenizer assets and itemic-token vocabulary expansion | [`README`](tokenizer/README.md) |
| `verl_distillation/` | On-policy distillation stage (forked from verl) | [`README`](verl_distillation/README.md) |
| `verl_rl/` | Reinforcement learning stage (forked from verl) | [`README`](verl_rl/README.md) |

> ★ The **data pipeline README** (`data/onerec_data/sft/cot/README.md`) is the main
> entry point — it walks through every step from raw parquet inputs to the final
> sharded SFT dataset, and points to the corresponding training recipe at each
> stage. Start there.

## 🚀 Quick Start

### 1. Data pipeline (produces the SFT training data)

See [`data/onerec_data/sft/cot/README.md`](data/onerec_data/sft/cot/README.md) for
the full end-to-end recipe. In short:

```bash
# from the repo root
cd data/onerec_data/sft/cot
bash 0_caption_summary/run_mpi.sh    # step 0 (optional): caption summary
bash 1_history_text/run_mpi.sh       # step 1: history + caption text
API_CONFIG_PATH=2_cot/api_config.json bash 2_cot/run.sh   # step 2: CoT synthesis
bash 3_ppl_eval/main_version/0_build_warmup_data.sh       # step 0': warm-up scorer data
# → train the 200-step warm-up scorer (see pretrain/README.md)
bash 3_ppl_eval/main_version/1_preprocess.sh              # step 3: PPL preprocess
bash 3_ppl_eval/main_version/2_eval_ppl.sh               # step 3: PPL eval
bash 3_ppl_eval/main_version/3_join.sh                   # step 3: join
bash 3_ppl_eval/main_version/4_filter.sh                 # step 4: annotate → p50
MAX_LOG_PPL_COT=<p50> bash 3_ppl_eval/main_version/4_filter.sh   # step 4: threshold
bash 3_ppl_eval/main_version/5_data_format.sh           # step 5: final SFT data
```

### 2. Training

See [`pretrain/README.md`](pretrain/README.md) for environment setup, vocabulary
expansion, and the training entrypoints. The two R²-SFT recipes are:

```bash
cd pretrain
# (a) 200-step warm-up scorer (used as the PPL scorer in step 3)
python3 auto_train/auto_train.py auto_train/r2sft_warmup_200steps.yaml

# (b) main target-aware SFT on PPL-filtered CoT
python3 auto_train/auto_train.py auto_train/r2sft_sft_ppl.yaml
```

### 3. Evaluation

See [`benchmarks/README.md`](benchmarks/README.md) for the RecIF-Bench evaluation
harness (Ray cluster + vLLM inference).

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 📋 Requirements

- Python 3.8+
- PyTorch (with FSDP)
- OpenMPI or compatible MPI implementation
- NCCL (for GPU communication)
- See `pretrain/README.md` and `benchmarks/README.md` for the full dependency list

## 🙏 Acknowledgements

This project builds on and benefits from the
[OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) project.
We thank the OpenOneRec team for releasing the pretraining, benchmark,
data-processing, tokenizer, and post-training components that form the
foundation of this work.

We also thank Qwen3 for the base model architecture, and the
[verl](https://github.com/volcengine/verl) team for the distributed
post-training infrastructure used in the RL and distillation stages.