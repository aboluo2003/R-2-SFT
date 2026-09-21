import torch
import torch.nn as nn
import torch.nn.functional as F
from onerec_llm.utils.time_tracker import TimeTracker
# ===================================================================
# Cross-Entropy Loss Function
# ===================================================================

class CrossEntropyLoss(nn.Module):
    """
    An efficient CrossEntropyLoss module that avoids redundant calculations.
    It first computes per-token losses and then manually applies the reduction.
    (Based on the user-provided, superior implementation).
    """
    def __init__(self,
                 ignore_index: int = -100,
                 return_token_loss: bool = False,
                 shift_labels: bool = True,
                 reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.return_token_loss = return_token_loss
        self.reduction = reduction
        self.shift_labels = shift_labels

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, loss_weight_chunk: torch.Tensor = None, tokenwise_loss_weight: torch.Tensor = None):
        """
        Args:
            logits (torch.Tensor): A single tensor of shape (..., vocab_size).
            labels (torch.Tensor): Ground truth labels.
        """
        if tokenwise_loss_weight is not None:
            loss_weight_chunk = tokenwise_loss_weight

        vocab_size = logits.shape[-1]

        if self.shift_labels:
          logits = logits[:, :-1, :]
          labels = labels[:, 1:]

        # Reshape for cross-entropy calculation
        logits_flat = logits.float().reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Step 1: Compute per-token loss. This is the base for all other calculations.
        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            reduction="none"
        )

        if loss_weight_chunk is not None:
          loss = (per_token_loss * loss_weight_chunk).sum()
        else:
          loss = per_token_loss.sum()
        if self.reduction == "mean":
            # Ensure we divide by the number of valid (non-ignored) tokens
            total_elements = (labels_flat != self.ignore_index).sum()
            if total_elements > 0:
                loss /= total_elements
            else: # Handle case where all tokens are ignored
                loss.zero_()

        # Return what's requested
        if self.return_token_loss:
            return loss, per_token_loss

        return loss

# ===================================================================
# Memory-Efficient Chunked Loss Computer
# ===================================================================

class ChunkedLossComputer:
    """
    内存高效的两阶段反向传播控制器

    【核心功能】
    解决大型语言模型(LLM)中lm_head过大导致的显存不足问题，通过分块计算实现内存优化。

    【实现原理】
    采用两阶段计算策略：
    1. 将输入序列分成多个小批次(minibatch)，对每个批次单独计算logits和损失。
    2. 手动计算每个批次的梯度并累加，而不是一次性计算所有梯度。
    3. 最后将累加的梯度应用到模型参数和输入上。

    【内存优化效果】
    通过分块处理，避免了一次性为整个序列分配巨大的中间张量，显著减少GPU内存使用峰值。

    注意：
    返回的两个loss都是bp过+detach过的
    请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!!
    请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!!
    请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!!
    """
    def __init__(self, lm_head: nn.Module, loss_fn: nn.Module, minibatch_size: int, shift_labels: bool = True):
        """
        初始化两阶段梯度计算器

        参数:
            lm_head: 语言模型的输出层，通常是nn.Linear。也可以是任意与loss_fn适配的nn.Module。
            loss_fn: 损失函数。该函数必须返回一个元组 (avg_loss, per_token_loss)。
            minibatch_size: 每个分块的大小，用于控制内存使用。
            shift_labels: 是否偏移标签(用于自回归模型)。
        """
        if not isinstance(lm_head, nn.Module) or not isinstance(loss_fn, nn.Module):
            raise TypeError("lm_head和loss_fn必须是nn.Module的实例")

        self.lm_head = lm_head
        self.loss_fn = loss_fn
        self.minibatch_size = minibatch_size
        self.shift_labels = shift_labels
        self.loss_info = {}
        self.ticker = TimeTracker()

    def forward_and_backward(self, input: torch.Tensor, labels: torch.Tensor, loss_fn_args: dict = {}, tokenwise_loss_weight=None,
                             main_loss_beta: float = 1.0):
        """
        执行两阶段的前向和反向传播过程

        参数:
            input: 输入张量，形状通常为[batch_size, seq_len, hidden_dim]。
            labels: 标签张量，形状通常为[batch_size, seq_len]。

        返回:
            tuple[torch.Tensor, torch.Tensor]:
                - final_avg_loss: 整个输入的平均损失值。
                - per_token_loss: 整个输入的per-token损失。

        注意：
        返回的两个loss都是bp过+detach过的
        请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!! 若有必要，请你把loss计算逻辑写到loss_fn中
        请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!! 若有必要，请你把loss计算逻辑写到loss_fn中
        请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!! 若有必要，请你把loss计算逻辑写到loss_fn中
        """
        self.ticker.tick("lm_head")
        # ═══════════════════════════════════════════════════════════════
        # lm_head 梯度由 loss.backward() 触发 FSDP hooks (reduce_scatter)
        # 自动管理，无需再手动 torch.autograd.grad() 和 p.grad = ...
        # ═══════════════════════════════════════════════════════════════
        grad_input_full = torch.zeros_like(input)

        total_loss_sum_for_reporting = torch.tensor(0.0, device=input.device)

        all_per_token_losses = []

        seq_len = input.size(1)

        # 计算总有效元素数量
        labels_to_count = labels[:, 1:] if self.shift_labels else labels
        total_elements = (labels_to_count != getattr(self.loss_fn, 'ignore_index', -100)).sum()
        total_elements_f = total_elements.float()

        # ── 第一阶段: 分块计算前向和梯度累加 ────────────────────────────
        # ⚠️ 关键约束: 每个 chunk 必须调用 loss_chunk_sum.backward()，
        # 无论该 chunk 是否有有效 token。原因是：
        #   1. lm_head 被 FSDP2 包裹，每次 backward() 都会触发 FSDP 的
        #      reduce-scatter（collective 通信，DP 组内同步）。
        #   2. 不同 rank 的 chunk 内有效 token 数可能不同（取决于数据分布），
        #      如果某 rank 在零有效 chunk 上跳过 backward()，则该 rank 的
        #      lm_head FSDP reduce-scatter 次数少于同组其他 rank。
        #   3. FSDP 内部状态计数错乱 → 下一轮 forward 的 all-gather 死锁。
        #
        # 解决方案：零有效 chunk 使用 (logits_chunk.sum() * 0.0) 作为 loss，
        # 它通过 lm_head 的计算图 → backward 时 FSDP hooks 正常触发，
        # 但梯度为 0，不影响最终梯度累加结果。
        # ═══════════════════════════════════════════════════════════════
        for i in range(0, seq_len, self.minibatch_size):
            start, end = i, min(i + self.minibatch_size, seq_len)
            # .detach() 切断与 transformer 计算图的连接，
            # requires_grad_() 创建新叶子节点，只包含本 chunk → lm_head → loss 的子图。
            input_chunk = input[:, start:end, :].detach().requires_grad_()

            if tokenwise_loss_weight is not None:
                assert tokenwise_loss_weight.shape == labels.shape, (
                    f"tokenwise_loss_weight.shape={tokenwise_loss_weight.shape}, "
                    f"labels.shape={labels.shape}"
                )
                loss_weight_chunk = tokenwise_loss_weight[:, start:end]
                loss_weight_chunk_flat = loss_weight_chunk.reshape(-1)
            else:
                loss_weight_chunk_flat = 1

            logits_chunk = self.lm_head(input_chunk)

            if self.shift_labels:
                label_start, label_end = start + 1, end + 1
                labels_chunk = labels[:, label_start:label_end]
                if logits_chunk.size(1) > labels_chunk.size(1):
                    logits_chunk = logits_chunk[:, :labels_chunk.size(1), :]
            else:
                labels_chunk = labels[:, start:end]

            # 边界 case: shift_labels 下尾部 chunk 可能无标签
            if labels_chunk.numel() == 0:
                loss_chunk_sum = (logits_chunk.sum() * 0.0)
                loss_chunk_sum.backward()
                grad_input_full[:, start:end, :] += input_chunk.grad  # grad=0，累加无害
                input_chunk.grad = None
                continue

            logits_flat = logits_chunk.reshape(-1, self.lm_head.out_features)
            labels_flat = labels_chunk.reshape(-1)

            loss_chunk_avg, per_token_loss_chunk = self.loss_fn(
                logits_flat, labels_flat,
                loss_weight_chunk=loss_weight_chunk_flat,
                **loss_fn_args,
            )

            ignore_index = getattr(self.loss_fn, 'ignore_index', -100)
            if isinstance(loss_weight_chunk_flat, torch.Tensor):
                valid_mask = (labels_flat != ignore_index) & (loss_weight_chunk_flat > 0)
            else:
                valid_mask = (labels_flat != ignore_index)
            valid_tokens_in_chunk = valid_mask.sum()

            if valid_tokens_in_chunk.item() == 0:
                # 零有效 token chunk: 用零 loss 保持 FSDP collective 对齐
                loss_chunk_sum = (logits_chunk.sum() * 0.0)
            else:
                loss_chunk_sum = main_loss_beta * (per_token_loss_chunk * valid_tokens_in_chunk).sum()
                total_loss_sum_for_reporting += (loss_chunk_avg * valid_tokens_in_chunk).detach()

            loss_chunk_sum.backward()
            grad_input_full[:, start:end, :] += input_chunk.grad
            input_chunk.grad = None

            all_per_token_losses.append(per_token_loss_chunk.detach())

        # ── 第二阶段: 应用累加的梯度 ──────────────────────────────────
        self.ticker.tick("llm")
        # 除以 total_elements 后显式 contiguous()，确保 cuBLAS 收到连续张量
        # total_elements==0 时使用 1.0 避免 NaN（此时 grad_input_full 全是零）
        divisor = total_elements_f if total_elements.item() > 0 else torch.tensor(1.0, device=input.device)
        grad_scaled = (grad_input_full / divisor).contiguous()
        input.backward(gradient=grad_scaled)
        self.ticker.tick("done")

        # 计算最终的平均损失
        if total_elements.item() > 0:
            final_avg_loss = (main_loss_beta * total_loss_sum_for_reporting / total_elements).detach()
        else:
            final_avg_loss = torch.tensor(0.0, device=input.device)
        per_token_loss = torch.cat(all_per_token_losses) if all_per_token_losses else None
        final_avg_loss.requires_grad = True

        self.loss_info = {
            'loss': final_avg_loss,
            'per_token_loss': per_token_loss
        }
        return final_avg_loss, per_token_loss
