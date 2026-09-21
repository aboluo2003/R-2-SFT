# Step 4: filter_ppl (PPL-gain-based RecIF CoT filtering)

Filters CoT samples by the PPL gap `diff = log_ppl_cot - log_ppl_x` (more negative = CoT is more helpful) computed in `step 3`, producing per-sample parquet that `step 5` consumes directly as `--cot_dir`.

> Conclusion based on `3_ppl_eval/3b_results.txt`: the SFT signal is required (pretrain has no predictive power for "which CoT is helpful"); gates `log_ppl_x < 9.0` + `diff < -0.05`.

## Data flow

```
step2 (2_cot)        step3 PPL eval (per-target)
  part-*.parquet       part-*.parquet (logp_y_given_x / logp_y_given_x_cot / num_target_tokens + render cols)
        |                        |
        +----------+-------------+
                   |
                   v
         3_ppl_eval/3_join_back.py (MPI, hash(sample_key)%W shard)
                   |
                   v
         joined_{pretrain|sft}_sid/  (per-target, restore step2 cols, drop render cols)
                   |
                   v
         4_filter_ppl/main.py (MPI, score -> gate -> merge -> optional top-k)
                   |
                   v
         4_filter_ppl/{pretrain|sft}/part-*.parquet  (per-sample, step5 consumes directly)
```

## Prerequisite: step 3 join (restore step2 columns dropped during PPL eval)

step3 PPL eval output is per-target and drops `history_captions / target_pids / target_captions` (needed by step5), keeping a bunch of render columns. `3_join_back.py` joins step2 with step3-eval by `sample_key`, outputting clean per-target parquet.

```bash
# pretrain
bash data/onerec_data/sft/cot/3_ppl_eval/3_run_join.sh
# sft
RUN_TYPE=sft bash data/onerec_data/sft/cot/3_ppl_eval/3_run_join.sh
```

join output columns: `ppl_sample_key, sample_key, uid, target_domain, target_idx, target_pid, cot, history_pids, history_captions, target_pids, target_captions, history_meta, logp_y_given_x, logp_y_given_x_cot, ppl_y_given_x, ppl_y_given_x_cot, num_target_tokens, eval_meta`.

## Step 4 usage

```bash
# sft (default gates: diff<-0.05 & log_ppl_x<9.0, merge=any)
bash data/onerec_data/sft/cot/4_filter_ppl/run_mpi_sft.sh

# specify hostfile
bash .../run_mpi_sft.sh /etc/mpi/hostfile

# top-k (among gated survivors, take top-N samples by aggregate diff ascending)
LIMIT=100000 bash .../run_mpi_sft.sh
LIMIT_RATIO=0.3 bash .../run_mpi_sft.sh

# adjust gates
MAX_DIFF=-0.10 MAX_LOG_PPL_X=8.0 MERGE_RULE=all bash .../run_mpi_sft.sh

# annotate_all (no filtering, just score + append filter_* cols; gates empty)
MAX_DIFF="" MAX_LOG_PPL_X="" bash .../run_mpi_sft.sh
```

## Metrics and gates

per-target:
- `log_ppl_x   = -logp_y_given_x     / num_target_tokens`
- `log_ppl_cot = -logp_y_given_x_cot / num_target_tokens`
- `diff        = log_ppl_cot - log_ppl_x` (more negative = CoT more helpful)

per-target decision:
- `valid`   = scoreable (both logp finite) AND `log_ppl_x < max_log_ppl_x` (hygiene gate, drop noisy targets)
- `helpful` = `valid` AND `diff < max_diff` (CoT effectiveness gate)

per-sample aggregation:
- `filter_score` = mean(diff over valid targets) (lower is better)
- survival by `--merge_rule`: `any` (default, any target passes) / `all` / `majority` / `mean` (aggregate score < max_diff)

modes:
| mode | trigger | filter_rank |
|------|------|-------------|
| `annotate_all` | no gate / limit | -1 |
| `threshold` | `--max_diff` / `--max_log_ppl_x` | -1 |
| `top_k` | `--limit` / `--limit_ratio` (can stack on top of gates) | 1=best |

## Output schema (per-sample, merged targets)

| column | description |
|----|------|
| `sample_key, uid, target_domain` | step2 original cols |
| `cot, history_pids, history_captions, target_pids, target_captions, history_meta` | step2 original cols, step5 consumes directly |
| `filter_score` | aggregate diff (mean over valid targets), lower is better |
| `filter_rank` | top-k global rank (1=best); threshold/annotate = -1 |
| `filter_meta` | JSON, per-target `target_idx/target_pid/log_ppl_x/log_ppl_cot/diff/num_tokens/valid/helpful` + stats, for step5 split ablation tracing |

`filter_*` cols are auto-ignored by step5 (it only reads step2 cols).

## MPI strategy

Shard by `hash(sample_key) % world_size`; each rank owns all targets of its samples, scoring / gating / merging locally. top-k mode allgathers lightweight candidates `(filter_score, sample_key)`, rank0 globally sorts and takes top-N then bcasts the kept set; each rank only writes its sharded hits, no cross-rank full-row transfer.

## Main parameters

| param / env var | default | description |
|----|----|------|
| `--max_diff` / `MAX_DIFF` | -0.05 | CoT effectiveness gate; empty = disabled |
| `--max_log_ppl_x` / `MAX_LOG_PPL_X` | 9.0 | hygiene gate; empty = disabled |
| `--max_log_ppl_cot` / `MAX_LOG_PPL_COT` | (empty) | target-SID PPL gate; empty = disabled |
| `--merge_rule` / `MERGE_RULE` | any | any/all/majority/mean |
| `--limit` / `LIMIT` | 0 | top-k k |
| `--limit_ratio` / `LIMIT_RATIO` | 0.0 | top-k ratio (overrides limit) |
| `--rows_per_output_file` / `ROWS_PER_OUTPUT_FILE` | 5000 | rows per output parquet |
| `SLOTS_PER_NODE` | 2 | MPI slots per node |
| `HOSTFILE` | /etc/mpi/hostfile | MPI hostfile |

Output dir includes `filter_summary.json` + console prints of gate / retained count / diff distribution.

## Handoff to step 5

```bash
COT_DIR=/path/to/4_filter_ppl/sft \
bash data/onerec_data/sft/cot/5_data_format/run_mpi.sh
```

step5 reads `sample_key, uid, target_domain, cot, history_pids, history_captions, target_pids, target_captions, history_meta` — all preserved by this step.

> split-target decisions are left to step5: this step's `filter_meta` keeps per-target diff, so step5 can decide whether to split a sample / keep only helpful targets.
