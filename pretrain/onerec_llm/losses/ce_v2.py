import torch
import torch.nn as nn
import torch.nn.functional as F
from onerec_llm.utils.time_tracker import TimeTracker
from typing import Optional
# ===================================================================
# Cross-Entropy Loss Function
# ===================================================================

# class CrossEntropyLoss(nn.Module):
#     """
#     An efficient CrossEntropyLoss module that avoids redundant calculations.
#     It first computes per-token losses and then manually applies the reduction.
#     (Based on the user-provided, superior implementation).
#     """
#     def __init__(self,
#                  ignore_index: int = -100,
#                  return_token_loss: bool = False,
#                  shift_labels: bool = True,
#                  reduction: str = "mean"):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.return_token_loss = return_token_loss
#         self.reduction = reduction
#         self.shift_labels = shift_labels

#     def forward(self, logits: torch.Tensor, labels: torch.Tensor):
#         """
#         Args:
#             logits (torch.Tensor): A single tensor of shape (..., vocab_size).
#             labels (torch.Tensor): Ground truth labels.
#         """
#         vocab_size = logits.shape[-1]
        
#         if self.shift_labels:
#           logits = logits[:, :-1, :]
#           labels = labels[:, 1:]

#         # Reshape for cross-entropy calculation
#         logits_flat = logits.float().reshape(-1, vocab_size)
#         labels_flat = labels.reshape(-1)

#         # Step 1: Compute per-token loss. This is the base for all other calculations.
#         per_token_loss = F.cross_entropy(
#             logits_flat,
#             labels_flat,
#             ignore_index=self.ignore_index,
#             reduction="none"
#         )
        
#         # Step 2: Manually apply reduction to get the final loss.
#         loss = per_token_loss.sum()
#         if self.reduction == "mean":
#             # Ensure we divide by the number of valid (non-ignored) tokens
#             total_elements = (labels_flat != self.ignore_index).sum()
#             if total_elements > 0:
#                 loss /= total_elements
#             else: # Handle case where all tokens are ignored
#                 loss.zero_()

#         # Return what's requested
#         if self.return_token_loss:
#             return loss, per_token_loss
        
#         return loss



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
            loss_weight_chunk =  tokenwise_loss_weight

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
          # print(f"per_token_loss.shape={per_token_loss.shape}, loss_weight_chunk.shape={loss_weight_chunk.shape}")
          # if torch.distributed.get_rank() == 0:
            # print(f"per_token_loss.shape={per_token_loss}, loss_weight_chunk.shape={loss_weight_chunk}")
            # if int(time.time() * 100) % 10 == 1:
            #   print_input_info(
            #     {
            #       "labels_flat": labels_flat,
            #       "loss_weight_chunk": loss_weight_chunk,
            #       "per_token_loss": per_token_loss,
            #     },
            #     save_path=f"debug/ce_loss_input_info_{time.time()}.pt"
            #   )
          per_token_loss = per_token_loss # * loss_weight_chunk
        
          # Step 2: Manually apply reduction to get the final loss.
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

# class ChunkedLossComputer:
#     """
#     Memory-efficient chunked loss computer for solving OOM issues caused by large lm_head in LLMs.

#     By computing the input sequence in chunks and manually accumulating gradients,
#     it avoids allocating huge intermediate tensors for the entire sequence at once.

#     Note: The returned loss has already been backpropagated and detached,
#     and cannot be used for operations requiring gradients.
#     """
#     def __init__(self, lm_head: nn.Module, loss_fn: nn.Module, minibatch_size: int, shift_labels: bool = True):
#         """
#         Initialize the chunked loss computer.

#         Args:
#             lm_head: The output layer of the language model (typically nn.Linear)
#             loss_fn: Loss function, must return (avg_loss, per_token_loss) tuple
#             minibatch_size: Size of each chunk, used to control memory usage
#             shift_labels: Whether to shift labels (for autoregressive models)
#         """
#         if not isinstance(lm_head, nn.Module) or not isinstance(loss_fn, nn.Module):
#             raise TypeError("lm_head and loss_fn must be instances of nn.Module")
            
#         self.lm_head = lm_head
#         self.loss_fn = loss_fn
#         self.minibatch_size = minibatch_size
#         self.shift_labels = shift_labels
#         self.loss_info = {}
#         self.ticker = TimeTracker()

#     def forward_and_backward(self, input: torch.Tensor, labels: torch.Tensor, loss_fn_args: dict = {}):
#         """
#         Execute chunked forward and backward propagation.

#         Args:
#             input: Input tensor with shape [batch_size, seq_len, hidden_dim]
#             labels: Label tensor with shape [batch_size, seq_len]
#             loss_fn_args: Additional arguments passed to the loss function

#         Returns:
#             tuple[torch.Tensor, torch.Tensor]: (final_avg_loss, per_token_loss)

#         Note: The returned loss has already been backpropagated and detached,
#         and cannot be used for operations requiring gradients.
#         """
#         self.ticker.tick("lm_head")
#         params = list(self.lm_head.parameters())
#         grad_accs = [torch.zeros_like(p) for p in params]
#         grad_input_full = torch.zeros_like(input)

#         total_loss_sum_for_reporting = torch.tensor(0.0, device=input.device)
#         all_per_token_losses = []

#         seq_len = input.size(1)
        
#         # Calculate total number of valid elements
#         labels_to_count = labels[:, 1:] if self.shift_labels else labels
#         total_elements = (labels_to_count != getattr(self.loss_fn, 'ignore_index', -100)).sum()
        
#         if total_elements.item() == 0:
#             return torch.tensor(0.0, device=input.device), None

#         # Chunked forward and gradient accumulation
#         for i in range(0, seq_len, self.minibatch_size):
#             start, end = i, min(i + self.minibatch_size, seq_len)
#             input_chunk = input[:, start:end, :].detach().requires_grad_()
            
#             logits_chunk = self.lm_head(input_chunk)

#             if self.shift_labels:
#                 label_start, label_end = start + 1, end + 1
#                 labels_chunk = labels[:, label_start:label_end]
#                 # Ensure logits and labels have matching lengths
#                 if logits_chunk.size(1) > labels_chunk.size(1):
#                     logits_chunk = logits_chunk[:, :labels_chunk.size(1), :]
#             else:
#                 labels_chunk = labels[:, start:end]

#             if labels_chunk.numel() == 0:
#                 continue

#             logits_flat = logits_chunk.reshape(-1, self.lm_head.out_features)
#             labels_flat = labels_chunk.reshape(-1)
            
#             # Compute loss
#             loss_chunk_avg, per_token_loss_chunk = self.loss_fn(logits_flat, labels_flat, **loss_fn_args)

#             # Convert to sum loss for backward propagation
#             valid_tokens_in_chunk = (labels_flat != getattr(self.loss_fn, 'ignore_index', -100)).sum()
            
#             if valid_tokens_in_chunk.item() == 0:
#                 all_per_token_losses.append(per_token_loss_chunk.detach())
#                 continue
            
#             loss_chunk_sum = loss_chunk_avg * valid_tokens_in_chunk

#             # Manually compute gradients and accumulate
#             tensors_to_grad = [p for p in params if p.requires_grad] + [input_chunk]
#             grads = torch.autograd.grad(outputs=loss_chunk_sum, inputs=tensors_to_grad, retain_graph=False)
        
#             grad_idx = 0
#             for j in range(len(params)):
#                 if params[j].requires_grad:
#                     grad_accs[j] += grads[grad_idx]
#                     grad_idx += 1
#             grad_input_full[:, start:end, :] = grads[grad_idx]

#             total_loss_sum_for_reporting += loss_chunk_sum.detach()
#             all_per_token_losses.append(per_token_loss_chunk.detach())
        
#         # Apply accumulated gradients
#         for j, p in enumerate(params):
#             if p.requires_grad:
#                 p.grad = grad_accs[j] / total_elements

#         self.ticker.tick("llm")        
#         input.backward(gradient=grad_input_full / total_elements)
#         self.ticker.tick("done")
        
#         final_avg_loss = (total_loss_sum_for_reporting / total_elements).detach()
#         per_token_loss = torch.cat(all_per_token_losses) if all_per_token_losses else torch.zeros(0, device=input.device, dtype=torch.float32)
#         final_avg_loss.requires_grad = True

#         self.loss_info = {
#             'loss': final_avg_loss,
#             'per_token_loss': per_token_loss
#         }
#         return final_avg_loss, per_token_loss





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
                             main_loss_beta: float = 1.0, aux_loss_specs=None):
        """
        执行两阶段的前向和反向传播过程

        参数:
            input: 输入张量，形状通常为[batch_size, seq_len, hidden_dim]。
            labels: 标签张量，形状通常为[batch_size, seq_len]。
            aux_loss_specs: 可选的辅助 loss 列表，每个 spec 是一个 dict：
                {
                    "mask": Tensor[B, L] bool,  # aux loss 位置（loss position，即 logit 所在位置）
                    "targets": Tensor[num_aux] int,  # 对应位置的目标 token id（按 mask 中 1 的顺序排列）
                    "beta": float,  # loss 权重
                    "alt_log_probs": Optional[Tensor[num_aux] float],  # 若提供，计算 margin loss
                        margin = clamp(alt_log_probs - main_log_probs, min=0).mean()
                        若不提供，计算 NLL loss = -main_log_probs.mean()
                }
                在 chunk 循环中 inline 计算，梯度通过 input_chunk → grad_input_full → input.backward()
                流回 transformer，无需额外 forward。

        返回:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - final_avg_loss: 整个输入的平均损失值（不含 aux loss，仅 main CE）。
                - per_token_loss: 整个输入的 per-token 损失。
                - aux_loss_sum: 所有 aux loss 的加权和（已 backward + detach，仅用于 logging）。

        注意：
            返回的 loss 都是 bp 过 + detach 过的，不可再用于需要 bp 的操作。
        """
        self.ticker.tick("lm_head")
        # Reset debug flag for each forward call
        self._debug_printed = False
        grad_input_full = torch.zeros_like(input)

        total_loss_sum_for_reporting = torch.tensor(0.0, device=input.device)
        all_per_token_losses = []

        seq_len = input.size(1)

        # 计算总有效元素数量
        labels_to_count = labels[:, 1:] if self.shift_labels else labels
        total_elements = (labels_to_count != getattr(self.loss_fn, 'ignore_index', -100)).sum()
        total_elements_f = total_elements.float()

        # ── 预处理 aux_loss_specs：构建全局位置 → spec 内索引的映射 ──
        # 同时预计算每个 spec 的 aux_scale = total_elements / num_aux_tokens
        # 用于在 chunk 循环中把 aux SUM 缩放到与 main CE 相同的 per-token mean 尺度
        aux_prepared = []
        if aux_loss_specs:
            for spec_idx, spec in enumerate(aux_loss_specs):
                mask = spec["mask"]
                beta = spec.get("beta", 1.0)
                targets = spec["targets"]
                alt_log_probs = spec.get("alt_log_probs", None)
                name = spec.get("name", str(spec_idx))
                # 全局 aux 位置（loss position），按顺序排列
                aux_positions = mask[0].nonzero(as_tuple=False).flatten()  # [num_aux]
                num_aux = len(aux_positions)
                # aux_scale: 把 aux SUM 乘以这个系数，使得经过 1/total_elements 缩放后
                # 等价于 1/num_aux 缩放（即 per-token mean aux loss）
                # aux_sum * aux_scale / total_elements = aux_sum / num_aux = mean_aux
                if num_aux > 0 and total_elements.item() > 0:
                    aux_scale = (total_elements_f / float(num_aux)).to(input.device)
                else:
                    aux_scale = torch.tensor(0.0, device=input.device)
                aux_prepared.append({
                    "name": name,
                    "positions": aux_positions,
                    "targets": targets,
                    "beta": beta,
                    "alt_log_probs": alt_log_probs,
                    "count": num_aux,
                    "aux_scale": aux_scale,
                    "reporting_sum": torch.tensor(0.0, device=input.device),  # 累加 beta * aux_sum（未缩放）
                })

        has_aux = len(aux_prepared) > 0

        # ── 第一阶段: 分块计算前向和梯度累加 ────────────────────────────
        # ⚠️ 关键约束: 每个 chunk 必须调用 loss_chunk_sum.backward()，
        # 无论该 chunk 是否有有效 token（FSDP collective 对齐）。
        for i in range(0, seq_len, self.minibatch_size):
            start, end = i, min(i + self.minibatch_size, seq_len)
            input_chunk = input[:, start:end, :].detach().requires_grad_()

            if tokenwise_loss_weight is not None:
                assert tokenwise_loss_weight.shape == labels.shape, (
                    f"tokenwise_loss_weight.shape={tokenwise_loss_weight.shape}, "
                    f"labels.shape={labels.shape}"
                )
                if self.shift_labels:
                    w_start, w_end = start + 1, end + 1
                else:
                    w_start, w_end = start, end
                loss_weight_chunk = tokenwise_loss_weight[:, w_start:w_end]
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
                grad_input_full[:, start:end, :] += input_chunk.grad
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

            # ── 计算 main CE loss sum ──
            if valid_tokens_in_chunk.item() == 0:
                main_loss_sum = (logits_chunk.sum() * 0.0)
            else:
                if isinstance(loss_weight_chunk_flat, torch.Tensor):
                    main_loss_sum = (per_token_loss_chunk * loss_weight_chunk_flat).sum()
                else:
                    main_loss_sum = per_token_loss_chunk.sum()
                total_loss_sum_for_reporting += (loss_chunk_avg * valid_tokens_in_chunk).detach()

            main_loss_sum = main_loss_beta * main_loss_sum

            # ── Inline 计算 aux loss（如果有 aux 位置落在当前 chunk） ──
            # 用 SUM（不是 mean）计算 per-token aux loss，乘以 aux_scale 后加入 loss_chunk_sum
            # 这样经过 1/total_elements 缩放后，aux 梯度 = beta * mean_over_aux_tokens(grad)
            # 与原始非 chunked 模式的 aux 梯度一致
            aux_loss_in_chunk = None
            if has_aux:
                aux_terms = []
                for spec in aux_prepared:
                    positions = spec["positions"]
                    if positions.numel() == 0:
                        continue
                    # 找到落在 [start, end) 范围内的 aux 位置（positions 是 sorted）
                    lo_idx = torch.searchsorted(positions, torch.tensor(start, device=positions.device))
                    hi_idx = torch.searchsorted(positions, torch.tensor(end - 1, device=positions.device), right=True)
                    if lo_idx >= hi_idx:
                        continue

                    chunk_aux_global = positions[lo_idx:hi_idx]  # [n] 全局位置
                    chunk_aux_local = chunk_aux_global - start  # [n] chunk 内位置
                    chunk_aux_targets = spec["targets"][lo_idx:hi_idx]  # [n]

                    # 从 logits_chunk 提取对应位置的 logits
                    aux_logits = logits_chunk[0, chunk_aux_local, :]  # [n, V]
                    aux_log_probs = F.log_softmax(aux_logits.float(), dim=-1).gather(
                        -1, chunk_aux_targets.unsqueeze(-1)
                    ).squeeze(-1)  # [n]

                    # ── DEBUG: 打印 aux 位置/目标/NLL 对齐信息（仅前几个 chunk）──
                    import os as _os
                    _debug_aux = _os.environ.get("DEBUG_AUX_LOSS", "0") == "1"
                    if _debug_aux and lo_idx < hi_idx and not getattr(self, "_debug_printed", False):
                        self._debug_printed = True
                        _rank = 0
                        try:
                            import torch.distributed as _dist
                            _rank = _dist.get_rank() if _dist.is_initialized() else 0
                        except Exception:
                            pass
                        if _rank == 0:
                            print(f"\n{'='*70}")
                            print(f"[AUX DEBUG] spec={spec['name']}, chunk=[{start},{end}), "
                                  f"num_aux_in_chunk={hi_idx-lo_idx}")
                            # 打印前 5 个 aux 位置的信息
                            for _i in range(min(5, hi_idx - lo_idx)):
                                _gpos = chunk_aux_global[_i].item()
                                _lpos = chunk_aux_local[_i].item()
                                _target = chunk_aux_targets[_i].item()
                                _logprob = aux_log_probs[_i].item()
                                _nll = -_logprob
                                # 对比 main CE label
                                _label = labels_flat[_lpos].item()
                                _is_itemic = 151669 <= _target <= 176246
                                _label_match = (_target == _label)
                                print(f"  aux_pos={_gpos}  target_id={_target}  "
                                      f"log_prob={_logprob:.4f}  NLL={_nll:.4f}  "
                                      f"itemic={_is_itemic}  main_label={_label}  "
                                      f"match={_label_match}")
                            if spec["alt_log_probs"] is not None:
                                _alt = spec["alt_log_probs"][lo_idx:hi_idx][:5]
                                print(f"  alt_log_probs[:5] = {_alt.tolist()}")
                                _main = aux_log_probs[:5]
                                _margin = torch.clamp(_alt - _main, min=0.0)
                                print(f"  main_log_probs[:5] = {_main.tolist()}")
                                print(f"  margin[:5] = {_margin.tolist()}")
                            print(f"{'='*70}\n", flush=True)

                    if spec["alt_log_probs"] is not None:
                        # Think (gap) loss: margin = clamp(alt - main, min=0)
                        # alt_lp 是 detached 的（stop gradient），梯度只通过 aux_log_probs 流回
                        alt_lp = spec["alt_log_probs"][lo_idx:hi_idx]  # [n] detached
                        margin = torch.clamp(alt_lp - aux_log_probs, min=0.0)
                        # SUM (not mean): 乘 aux_scale 后等价于 mean
                        aux_sum = spec["beta"] * margin.sum()
                    else:
                        # Target loss: NLL = -log_probs
                        aux_sum = spec["beta"] * (-aux_log_probs.sum())

                    # 缩放：aux_sum * aux_scale / total_elements = aux_sum / num_aux = mean_aux
                    aux_terms.append(aux_sum * spec["aux_scale"])
                    # 用于 reporting：累加未缩放的 beta * aux_sum
                    spec["reporting_sum"] += (aux_sum).detach()

                if aux_terms:
                    aux_loss_in_chunk = torch.stack(aux_terms).sum()

            # ── 合并 main + aux loss 并 backward ──
            if aux_loss_in_chunk is not None:
                loss_chunk_sum = main_loss_sum + aux_loss_in_chunk
            else:
                loss_chunk_sum = main_loss_sum

            loss_chunk_sum.backward()
            grad_input_full[:, start:end, :] += input_chunk.grad
            input_chunk.grad = None

            all_per_token_losses.append(per_token_loss_chunk.detach())

        # ── 第二阶段: 应用累加的梯度 ──────────────────────────────────
        self.ticker.tick("llm")
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

        # 计算每个 spec 的 aux loss mean（用于 reporting）
        # reporting_sum = sum over chunks of (beta * per_token_aux_sum_in_chunk)
        # mean = reporting_sum / num_aux_tokens
        aux_loss_by_spec = {}
        for spec in aux_prepared:
            if spec["count"] > 0:
                aux_loss_by_spec[spec["name"]] = (spec["reporting_sum"] / spec["count"]).detach()
            else:
                aux_loss_by_spec[spec["name"]] = torch.tensor(0.0, device=input.device)

        self.loss_info = {
            'loss': final_avg_loss,
            'per_token_loss': per_token_loss,
            'aux_loss_by_spec': aux_loss_by_spec,
        }
        return final_avg_loss, per_token_loss, aux_loss_by_spec


# ===================================================================
# Soft-Label Cross-Entropy Loss for itemic (VQ codebook) tokens
# ===================================================================

class SoftLabelCELoss(nn.Module):
    """Cross-entropy loss with soft-label support for itemic tokens.

    For **text** positions: standard hard CE loss (same as CrossEntropyLoss).
    For **itemic** positions: the LLM's original vocabulary logits are masked
    to ``-inf`` so only codebook entries carry probability; the target is a
    soft distribution over ``codebook_size`` entries computed from z_e vs
    projected codebook similarity scaled by temperature.

    Loss at itemic positions::

        log_p = log_softmax(logits[mask, cb_offset:cb_offset+cb_size])
        loss  = -Σ soft_label * log_p   (per position)

    Args:
        ignore_index:     Token index to ignore in loss (default -100).
        return_token_loss: Whether to return per-token loss tensor.
        shift_labels:     Whether to shift labels for autoregressive models.
        reduction:        "mean" or "sum".
        codebook_offset:  Vocab index offset for codebook entries
                          (default 1, matching ``code_idx + 1`` convention).
        codebook_size:    Number of codebook entries per level.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        return_token_loss: bool = True,
        shift_labels: bool = True,
        reduction: str = "mean",
        codebook_offset: int = 1,
        codebook_size: int = 8192,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.return_token_loss = return_token_loss
        self.shift_labels = shift_labels
        self.reduction = reduction
        self.codebook_offset = codebook_offset
        self.codebook_size = codebook_size

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_labels: Optional[torch.Tensor] = None,
        itemic_id_mask: Optional[torch.Tensor] = None,
        tokenwise_loss_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            logits:              (B, L, vocab_size) LLM output logits.
            labels:              (B, L) hard labels (used for text positions).
            soft_labels:         (B, L, codebook_size) soft distributions
                                 for itemic positions; zero at text positions.
            itemic_id_mask:      (B, L) >0 means itemic; value = level index.
            tokenwise_loss_weight: (flat or B×L) per-token weight multiplier.
        """
        if tokenwise_loss_weight is not None and soft_labels is None:
            # Fall back to standard CE when no soft labels are provided
            loss_weight = tokenwise_loss_weight
            vocab_size = logits.shape[-1]
            if self.shift_labels:
                logits = logits[:, :-1, :]; labels = labels[:, 1:]
            logits_flat = logits.float().reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            per_token_loss = F.cross_entropy(
                logits_flat, labels_flat,
                ignore_index=self.ignore_index, reduction="none"
            )
            if loss_weight is not None:
                loss = (per_token_loss * loss_weight.reshape(-1)).sum()
            else:
                loss = per_token_loss.sum()
            if self.reduction == "mean":
                total_elements = (labels_flat != self.ignore_index).sum()
                loss = loss / total_elements if total_elements > 0 else loss.zero_()
            return (loss, per_token_loss) if self.return_token_loss else loss

        vocab_size = logits.shape[-1]
        cb_start = self.codebook_offset
        cb_end = cb_start + self.codebook_size

        if self.shift_labels:
            logits = logits[:, :-1, :]; labels = labels[:, 1:]
            if soft_labels is not None:  soft_labels = soft_labels[:, 1:, :]
            if itemic_id_mask is not None: itemic_id_mask = itemic_id_mask[:, 1:]
            if tokenwise_loss_weight is not None:
                tokenwise_loss_weight = tokenwise_loss_weight[:, 1:]

        logits_flat = logits.float().reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # ── Standard CE for ALL positions (used for text) ────────────────
        per_token_loss = F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=self.ignore_index, reduction="none"
        )

        # ── Override itemic positions with soft CE loss ──────────────────
        if soft_labels is not None and itemic_id_mask is not None:
            itemic_flat = (itemic_id_mask.reshape(-1) > 0)
            if itemic_flat.any():
                sl_flat = soft_labels.reshape(-1, self.codebook_size)
                il = logits_flat[itemic_flat]                  # (N_it, V)
                # Mask non-codebook logits → -inf
                mask = torch.full_like(il, float('-inf'))
                mask[:, cb_start:cb_end] = 0.0
                il_masked = il + mask
                log_p = F.log_softmax(il_masked, dim=-1)
                log_p_cb = log_p[:, cb_start:cb_end]           # (N_it, K)
                sl_it = sl_flat[itemic_flat]                   # (N_it, K)
                soft_ce = -(sl_it * log_p_cb).sum(dim=-1)     # (N_it,)
                per_token_loss[itemic_flat] = soft_ce

        if tokenwise_loss_weight is not None:
            w = tokenwise_loss_weight.reshape(-1) if tokenwise_loss_weight.ndim >= 2 else tokenwise_loss_weight
            loss = (per_token_loss * w).sum()
        else:
            loss = per_token_loss.sum()
        if self.reduction == "mean":
            total_elements = (labels_flat != self.ignore_index).sum()
            loss = loss / total_elements if total_elements > 0 else loss.zero_()

        return (loss, per_token_loss) if self.return_token_loss else loss