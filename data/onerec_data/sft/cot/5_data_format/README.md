# Step 5: data_format (RecIF CoT SFT 数据集合成)

把 step 2 (`2_cot`) 或 step 4 (`4_filter`) 产出的 CoT 样本，与从 `onerec_bench_release.parquet` 构造的非 CoT 推荐样本、以及 `caption2sid` / `sid2caption` 任务样本，合成统一的 SFT 训练格式，做全局 shuffle + 分片，并生成 `file_list.json`。

## 数据流

```
输入                                          输出
─────────────────────────────────             ─────────────────────────────────
step2 (2_cot) / step4 (4_filter)    ─┐
  part-*.parquet                     │  阶段1: MPI 并行构造 + 临时文件
    {uid, target_domain, cot,         │
     history_pids, history_captions,  │     ┌──────────────────────────┐
     target_pids, ...}                ├────▶│ rank r 构造本地样本      │
                                      │     │ → temp_rank{r}.parquet   │
onerec_bench_release.parquet         │     └──────────────────────────┘
  (非 CoT video/ad/product)           │            │
                                      │            ▼
caption shards (pid+dense_caption)   │     Allgather (path, row_count)
  (sid2caption / caption2sid)        │            │
                                      │            ▼
pid2sid / product_pid2sid             │     阶段2: 确定性置换 + 并行 shard 写出
                                      │     ┌──────────────────────────┐
                                      └────▶│ 全局 perm (相同 seed)     │
                                            │ rank r 拥有 [r*k,(r+1)*k) │
                                            │ 反查 (temp_file, local_idx)│
                                            │ 从共享 FS 读 → 写出 shard  │
                                            └──────────────────────────┘
                                                      │
                                                      ▼
                                            output_dir/part-{shard:05d}.parquet
                                            blended/<name>.json  (file_list.json)
```

## 输出 schema

每个分片 parquet 的列：

| 列         | 类型   | 说明                                       |
|------------|--------|--------------------------------------------|
| `uuid`     | str    | 每条样本的唯一 id                           |
| `source`   | str    | 任务来源标识（见下表）                      |
| `messages` | str    | JSON 字符串，与 `sft/*.py` 格式对齐         |
| `metadata` | str    | JSON 字符串，含 `uid` / `task_type` 等元信息 |

### `source` 取值

| source                    | 任务                    | assistant 格式                                   |
|---------------------------|-------------------------|--------------------------------------------------|
| `RecIF_RecCOT`            | 推荐 + CoT（video/ad/product 统一）| ` {cot}` + `\n\n` + `{target_sids}`          |
| `RecIF_VideoRec`          | video 推荐（非 CoT）     | `{target_sids}`                                  |
| `RecIF_AdRec`             | ad 推荐（非 CoT）        | `{target_sids}`                                  |
| `RecIF_ProductRec`        | product 推荐（非 CoT）   | `{target_sids}`                                  |
| `RecIF_CaptionToSid`      | caption → sid            | `{target_sid}`                                   |
| `RecIF_ItemUnderstand`    | sid → caption            | `{caption}`                                      |

> **CoT source 统一**：video / ad / product 三个域的 CoT 样本都使用 `RecIF_RecCOT`，不区分域。
> 训练侧 `target_loss_sources_name` 只需列 `RecIF_RecCOT` 即可让所有 CoT 样本参与 target loss。

> 训练侧 `qwen3_dataset.py` 的 `add_think_pattern=True` 会自动检测 assistant content 里的  tag：
> - 有非空 think body → 给 user 加 `/think`
> - 无 think tag → 自动包空  并给 user 加 `/no_think`
>
> 因此本脚本对 CoT 样本输出 ` ... {cot} ... {target_sids}`，对非 CoT 样本直接输出 `{target_sids}`（训练侧自动处理）。

## Prompt 对齐

所有 prompt / system message 直接拷贝自：
- `data/onerec_data/sft/video_rec.py` (video rec, cot & non-cot)
- `data/onerec_data/sft/ad_rec.py` (ad rec, cot & non-cot)
- `data/onerec_data/sft/product_rec.py` (product rec, non-cot only)
- `data/onerec_data/sft/caption_to_sid.py` (caption2sid)
- `data/onerec_data/sft/item_understand.py` (sid2caption)

**ad rec 的两段式 history**：`ad_rec.py` 的 user prompt 是
```
用户观看过的视频：{video_sids}
用户点击过的广告视频：{ad_sids}
请根据...预测...广告视频。
```
CoT 样本的 history 优先从 bench row 读取**原始** pid（未经 caption 过滤），再按 `hist_longview_video_list[-100:]` + `hist_ad_pid[-200:]` 截断 + pid2sid 过滤，与非 CoT 样本完全对齐。仅在 bench row 缺失（uid 不在 bench）时退化为使用 step2 输出的 caption-filtered `history_pids`（按 `history_meta` 的 `kept_video_hist_len` / `kept_ad_hist_len` 切分两段）。

## CoT 与非 CoT 历史对齐（bug 修复）

**问题**：step1 (`1_history_text`) 用 caption 可用性过滤 history / target pid，step5 直接用这些 caption-filtered pid 构造 CoT 样本的 SID；非 CoT 样本则从 bench 读取原始 pid 经 pid2sid 过滤。caption 覆盖 ≠ pid2sid 覆盖 → 同一用户的 CoT 与非 CoT 历史序列不一致。

**修复**：step5 加载 bench 后构建 `{uid: row_idx}` 映射，CoT 样本按 uid 查找 bench row，用**原始** history / target pid（与非 CoT 同源），再统一截断 + pid2sid 过滤。caption 过滤丢弃的 pid（只要有 sid）也会出现在最终序列中。

**metadata 分段标记**：ad 域 CoT / 非 CoT 样本的 `metadata` JSON 含：
- `video_hist_sid_count`：「用户观看过的视频」段的 sid 个数
- `ad_hist_sid_count`：「用户点击过的广告视频」段的 sid 个数
- `history_source`：`"bench_original"`（正常路径）或 `"caption_filtered"`（退化路径）

input 中前 `video_hist_sid_count` 个 sid 属于视频段，后续 `ad_hist_sid_count` 个属于广告段。video 域 `ad_hist_sid_count=0`。

## ad domain CoT

`2_cot/run.sh` handles both video and ad in one pass:
- video samples use `2_cot/prompt.txt` (single-segment history)
- ad samples use `2_cot/prompt_ad.txt` (two-segment history: watched videos + clicked ads)

`2_cot/main.py` dispatches to `build_history_text` (video) or `build_history_text_ad`
(ad) based on `target_domain`. No separate ad-only run or merge step is needed.

## 用户重叠逻辑

user 单位 = `(uid, target_domain)`（同一 uid 在 video 域和 ad 域视为不同 user）。

- `ALLOW_COT_NON_COT_USER_OVERLAP=0`（默认）：非 CoT 推荐样本仅使用 CoT 用户集合以外的 `(uid, domain)`。product 域无 CoT，全部保留。
- `ALLOW_COT_NON_COT_USER_OVERLAP=1`：非 CoT 使用全部用户（含 CoT 用户），不做 dedup。训练时同一输入会出现两次（一次有 CoT、一次无 CoT），可同时学推理和直接预测。

## 全局 shuffle 的 MPI 实现

核心思路：**两阶段 + 确定性置换**，避免 Alltoall 传大数据。

### 阶段 1：per-rank 处理 + 临时文件

每个 rank round-robin 分配输入分片（CoT shards / caption shards），独立构造样本并写到共享 FS 上的 `temp_rank{r:05d}.parquet`。bench parquet 是单文件 1 row group，每 rank 全量读再按 uid 索引 round-robin（同节点 OS page cache 让磁盘只真正读一次），与 `1_history_text/main.py` 策略一致。

### Allgather 布局

`comm.allgather((temp_path, row_count))` 交换元数据。每 rank 都知道全局有 N 个临时文件、第 i 个文件有 $n_i$ 行、总共 $N_{total}$ 行。**不传样本数据，只传路径+行数。**

### 阶段 2：确定性置换 + 并行 shard 写出

1. **每 rank 用相同 seed** 调 `numpy.random.default_rng(seed).permutation(N_total)`，得到相同的全局置换 `perm`。
2. **每 rank 拥有连续的一段输出分片** `[r*k, (r+1)*k)`（round-robin 分配 shard idx）。
3. 通过 `perm[lo:hi]` 反查该分片需要哪些全局行索引，再映射到 `(temp_file_idx, local_row_idx)`。
4. 用 `pyarrow.Table.take(local_indices)` 从共享 FS 的临时文件读取所需行，写出 `part-{shard:05d}.parquet`。

**优点**：
- 无需 Alltoall 传大数据（`messages` 字段是长文本，序列化开销大）
- 共享 FS 随机读几十万行 parquet 很快（pyarrow 支持行级 take）
- 避免 rank 间内存倾斜
- `sample_ratio` 在生成 `perm` 后取前 `int(N_total * ratio)` 个索引，同样用固定 seed 保证所有 rank 一致

### 跳过 shuffle 的快速路径

`GLOBAL_SHUFFLE=0` 且 `SAMPLE_RATIO=1.0` 时，`perm = np.arange(N_total)`，阶段 2 退化为「每 rank 直接把自己的临时文件按行序重切为最终分片」，无跨 rank 读。

## 用法

### MPI 并行（服务器）

```bash
# 默认参数
bash data/onerec_data/sft/cot/5_data_format/run_mpi.sh

# 指定 hostfile
bash data/onerec_data/sft/cot/5_data_format/run_mpi.sh /etc/mpi/hostfile

# 自定义输入/输出
COT_DIR=/path/to/step4_output \
BENCH_PARQUET=/path/to/onerec_bench_release.parquet \
CAPTION_DIR=/path/to/caption_shards \
OUTPUT_DIR=/path/to/output \
BLENDED_JSON_NAME=recif_cot_sft_v1.json \
GLOBAL_SHUFFLE=1 SAMPLE_RATIO=0.8 \
SID2CAPTION_SIZE=260000 CAPTION2SID_SIZE=260000 \
bash data/onerec_data/sft/cot/5_data_format/run_mpi.sh

# 允许 CoT / 非 CoT 用户重叠
ALLOW_COT_NON_COT_USER_OVERLAP=1 \
bash data/onerec_data/sft/cot/5_data_format/run_mpi.sh
```

### 单进程（调试）

```bash
python3 data/onerec_data/sft/cot/5_data_format/main.py \
    --cot_dir /path/to/step2_output \
    --bench_parquet /path/to/onerec_bench_release.parquet \
    --caption_dir /path/to/caption_shards \
    --pid2sid /path/to/video_ad_pid2sid.parquet \
    --product_pid2sid /path/to/product_pid2sid.parquet \
    --output_dir /path/to/output \
    --blended_json_path /path/to/blended/recif_cot_sft.json \
    --global_shuffle 1 \
    --sample_ratio 1.0 \
    --sid2caption_size 260000 \
    --caption2sid_size 260000 \
    --seed 42
```

## 主要参数

| 参数 / 环境变量                          | 默认值      | 说明                                       |
|------------------------------------------|------------|--------------------------------------------|
| `--cot_dir` / `COT_DIR`                   | step2 输出 | CoT 输入目录（step2 或 step4 输出）          |
| `--bench_parquet` / `BENCH_PARQUET`       | bench     | `onerec_bench_release.parquet` 路径          |
| `--caption_dir` / `CAPTION_DIR`           | shards    | caption 分片目录（`pid + dense_caption`）    |
| `--pid2sid` / `PID2SID`                   | -         | `video_ad_pid2sid.parquet`                   |
| `--product_pid2sid` / `PRODUCT_PID2SID`   | -         | `product_pid2sid.parquet`                    |
| `--output_dir` / `OUTPUT_DIR`             | -         | 输出分片目录                                  |
| `--blended_json_path` / `BLENDED_JSON_PATH` | -       | `file_list.json` 路径（绝对路径）              |
| `--allow_cot_non_cot_user_overlap`         | 0         | 1=允许 CoT / 非 CoT 用户重叠                   |
| `--enable_cot_split_filter` / `ENABLE_COT_SPLIT_FILTER` | 1 | 1=通过 uid join bench split 列过滤 CoT（跳过 split!=0 的 uid） |
| `--global_shuffle` / `GLOBAL_SHUFFLE`     | 1         | 1=全局 shuffle                                |
| `--sample_ratio` / `SAMPLE_RATIO`         | 1.0       | 全局采样比例 (0, 1]                            |
| `--rows_per_shard` / `ROWS_PER_SHARD`     | 1000      | 每分片行数                                    |
| `--sid2caption_size` / `SID2CAPTION_SIZE` | 260000    | sid2caption 目标样本数                         |
| `--caption2sid_size` / `CAPTION2SID_SIZE` | 260000    | caption2sid 目标样本数                         |
| `--seed` / `SEED`                         | 42        | 随机种子                                      |
| `SLOTS_PER_NODE`                          | 2         | 每节点 MPI slot 数                            |
| `HOSTFILE`                                | /etc/mpi/hostfile | MPI hostfile                          |

## 注意事项

### 1. step 4 未完成时

step 4 (`4_filter`) 尚未实现，当前直接用 step 2 (`2_cot`) 的输出作为 `--cot_dir`。
等 step 4 完成后，把 `--cot_dir` 指向 step 4 的输出即可（字段名一致）。

### 2. `history_meta` 字段

ad CoT 样本的两段式 history 还原依赖 `history_meta` 里的 `field` / `kept_video_hist_len` / `kept_ad_hist_len`。
**step 2 当前的输出是否保留 `history_meta` 需要确认**（`2_cot/main.py` 的 `OUTPUT_COLUMNS` 包含此字段，实际数据中应该有）。
若 `history_meta` 缺失，ad CoT 会退化为单段 history（全部当 ad click history），prompt 仍可用但结构与 `ad_rec.py` 的两段式略有差异。

### 3. caption 仅覆盖 video/ad 域

`pid2caption.parquet` 只包含 video/ad 域的 pid，product 域的电商商品 caption 来自独立的 `goods_pid2caption`（列 `new_pid` + `caption`）。
- **CoT 链路**：product 域 CoT 的商品段 + target caption 用 `goods_pid2caption`（step1 `--goods_caption_uri`），视频段仍用 `pid2caption`。
- **caption2sid / sid2caption 任务**：仅基于 `video_ad_pid2sid.parquet` 构造，不含 product（这两个任务是 sid↔caption 基础理解任务，用 video/ad 域即可）。

### 4. split=0 过滤

**非 CoT 推荐样本**（video / ad / product）只处理 `onerec_bench_release.parquet` 中 `split=0` 的行，与 `video_rec.py` / `ad_rec.py` / `product_rec.py` 一致。

**CoT 样本**（来自 step2/step4）本身不含 `split` 字段（step1 `1_history_text` 不按 split 过滤）。本脚本通过 **uid join bench 的 `split` 列** 来过滤：
1. 从 bench 读取 `uid` + `split` 两列，构造 `{uid: split}` 映射（只读两列，开销很小）
2. 加载 CoT 样本时，跳过 `uid_to_split[uid] != 0` 的样本

这样 CoT 和非 CoT 都只使用 `split=0` 的用户，保持一致。

可通过 `--enable_cot_split_filter 0` 关闭此过滤（默认开启）。

> 当前 bench parquet 中 `split` 全部为 0（162074 行），过滤暂无实际影响，但保留以备将来 split 划分。

### 5. pid2sid 内存

`video_ad_pid2sid.parquet` 约 1588 万条，每 rank 全量加载成 dict 约占 2GB 内存。
16 ranks/节点 × 2GB = 32GB，需确保节点内存 ≥ 64GB。
同节点 OS page cache 让磁盘只真正读一次，但每 rank 的 Python dict 是独立副本。

### 6. file_list.json 路径

`file_list.json` 默认保存到 `${PREFIX}/pretrain/examples/blended/${BLENDED_JSON_NAME}`。
文件内为输出分片的**绝对路径**。训练时 `dataset_config` 指向此 json 即可。

### 7. 临时文件

阶段 1 的临时文件默认在 `${OUTPUT_DIR}/_temp/`。完成后 rank 0 自动清理（`KEEP_TEMP=0`）。
设 `KEEP_TEMP=1` 可保留用于调试。

### 8. viewfs / hdfs 支持

输入路径支持 `viewfs://` / `hdfs://` 前缀（需设置 `JAVA_HOME` / `HADOOP_HOME`）。
脚本会自动设置 `LD_LIBRARY_PATH` / `CLASSPATH` / `HADOOP_USER_NAME`，与 `reason/mpi/cross_reco/eval/2_filter/run_mpi.sh` 对齐。

## 与训练的衔接

训练侧 `dataset_config` 指向生成的 `file_list.json`：

```json
{
  "data_path": ".../pretrain/examples/blended/r2sft_sft_ppl.json",
  ...
}
```

`qwen3_dataset.py` 的关键参数：
- `add_think_pattern: true` — 自动处理  tag + `/think`/`/no_think`
- `think_loss_sources_name` — 指定哪些 `source` 参与 think loss（如 `RecIF_RecCOT`，所有 CoT 样本统一使用此 source）

## 后续（step 4 完成后）

1. 实现 `4_filter/main.py` + `run_mpi.sh`（按评分筛选 CoT）
2. 把 `--cot_dir` 指向 step 4 输出
3. step 4 输出 schema 需保留 `history_meta` 字段（与 step 2 一致），否则 ad CoT 的两段式 history 还原会退化
