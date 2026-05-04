from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        loss = (
            masked_mean(-per_token_logps, loss_mask, dim=None)
            if self.token_level_loss
            else masked_mean(-per_token_logps, loss_mask, dim=-1).mean()
        )

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        vllm_is_correction_type: str = "tis",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type

        # GSPO requires sequence-level loss
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if self.vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self.vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
            )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.policy_loss_type == "ppo":
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = log_probs - old_log_probs
            ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            log_ratio = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
                vllm_is = torch.exp(log_ratio).detach()
                mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
                vllm_is = vllm_is * mask
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                # seq-mask-tis: use sequence-level geometric mean only for filtering,
                # correction coefficients still use TIS (token-level clamp)
                seq_log_ratio = masked_mean(log_ratio, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
                vllm_is = torch.exp(log_ratio).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:
                # TIS: token-level clamp with low and high thresholds
                vllm_is = torch.exp(log_ratio).clamp(min=low_threshold, max=high_threshold).detach()
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
        )
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
        )
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask].squeeze(1)
        labels = labels[placeholder_mask]
        

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

class CRMLoss(nn.Module):
    """
    Conditional Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None, neg_weight: float = 1.0):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids
        self.pos_token_id = reward_token_ids[0] if reward_token_ids is not None else None
        self.neg_token_id = reward_token_ids[1] if reward_token_ids is not None else None
        self.eps = 1e-5
        self.neg_weight = neg_weight
        
    def forward(self, inputs: torch.Tensor, values: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        B, L = values.shape
       
        placeholder_mask = inputs == self.placeholder_token_id
        
        is_wrong = placeholder_mask & (labels == self.neg_token_id)
        
        indices = torch.arange(L, device=inputs.device).expand(B, L)

        last_placeholder_indices = (indices * placeholder_mask.long()).max(dim=1).values

        # 2. 提取最后一步的 label
        # 结果形状为 [Batch]
        last_step_labels = labels.gather(1, last_placeholder_indices.unsqueeze(1)).squeeze(1)

        # 3. 判定是否为负样本
        # 只有当样本包含占位符，且最后一个占位符对应的 label 是 neg_token 时，才算负样本
        has_placeholder = placeholder_mask.any(dim=1)
        is_negative_sample = has_placeholder & (last_step_labels == self.neg_token_id)
        # print(is_negative_sample)
        # 4. 判定是否为正样本
        is_positive_sample = has_placeholder & ~is_negative_sample
        # print(is_positive_sample)
        # print(is_negative_sample)
        # is_negative_sample = is_wrong.any(dim=1)
        # is_positive_sample = placeholder_mask.any(dim=1) & ~is_negative_sample
        

        log_h = F.logsigmoid(values)             
        log_1_m_h = F.logsigmoid(-values)           
        log_1_m_h_masked = log_1_m_h * placeholder_mask.float()
       
        # 公式 11: L_S = -sum(log(1 - h(t)))
        L_S_batch = -log_1_m_h_masked.sum(dim=1) 
        # print(L_S_batch)
        # quit()
        # 公式 12: L_W
        # S_T_batch = exp(sum(log(1 - h(t))))
        S_T_batch = torch.exp(log_1_m_h_masked.sum(dim=1))
        L_W_batch = -torch.log(torch.clamp(1.0 - S_T_batch, min=self.eps))

        # 公式 13: L_z
        z_indices = is_wrong.float().argmax(dim=1) 
        batch_idx = torch.arange(B, device=values.device)
        
        # 提取第一个出错位置的 log(h(z_i))
        log_h_zi = log_h[batch_idx, z_indices] 

        pos = torch.arange(L, device=values.device).unsqueeze(0).expand(B, L)
        before_zi_mask = placeholder_mask & (pos < z_indices.unsqueeze(1))
        
        # 累加 z_i 之前的 log(1 - h(t))
        sum_log_1_m_h_before_zi = (log_1_m_h * before_zi_mask.float()).sum(dim=1) 

        L_z_batch = -(log_h_zi + sum_log_1_m_h_before_zi) 

        # 整合 Loss
        total_loss_batch = torch.zeros(B, device=values.device)

        total_loss_batch[is_positive_sample] = L_S_batch[is_positive_sample]
        total_loss_batch[is_negative_sample] = self.neg_weight * (L_W_batch[is_negative_sample] + L_z_batch[is_negative_sample])

        valid_samples_count = is_positive_sample.sum() + is_negative_sample.sum()
        loss = total_loss_batch.sum() / torch.clamp(valid_samples_count.float(), min=1.0)
        loss_neg = total_loss_batch[is_negative_sample].sum() / torch.clamp(is_negative_sample.sum().float(), min=1.0)
        loss_pos = total_loss_batch[is_positive_sample].sum() / torch.clamp(is_positive_sample.sum().float(), min=1.0)
        if return_acc:
            probs = torch.sigmoid(values)
    
            # 2. 提取掩码和标签
            is_neg_step = placeholder_mask & (labels == self.neg_token_id)
            is_pos_step = placeholder_mask & (labels == self.pos_token_id)
            
            # 3. 计算各部分的准确率张量
            # 负样本：prob 越高越对；正样本：(1-prob) 越高越对
            acc_neg_tensor = probs[is_neg_step]
            acc_pos_tensor = 1.0 - probs[is_pos_step]
            
            # 4. 计算各部分数量
            n_neg = is_neg_step.sum().float()
            n_pos = is_pos_step.sum().float()
            n_total = n_neg + n_pos
            
            # 5. 指标汇总
            # 总体准确率 (Total Acc) - 反应全局拟合情况
            acc = (acc_neg_tensor.sum() + acc_pos_tensor.sum()) / torch.clamp(n_total, min=1.0)
            
            # 拆分准确率 - 反应预测偏好
            acc_neg = acc_neg_tensor.sum() / torch.clamp(n_neg, min=1.0)
            acc_pos = acc_pos_tensor.sum() / torch.clamp(n_pos, min=1.0)
            
            # 6. 计算“平衡准确率” (Balanced Acc) 
            # 如果正负样本极度不平衡，这个指标比 total_acc 更能反映真实能力
            # balanced_acc = (acc_neg + acc_pos) / 2.0
            
        else:
            acc = torch.tensor(0.0, device=values.device)
            acc_neg = torch.tensor(0.0, device=values.device)
            acc_pos = torch.tensor(0.0, device=values.device)
        return loss,loss_pos,loss_neg, acc, acc_pos, acc_neg


class TACRMLoss(nn.Module):
    """
    Conditional Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None, neg_weight: float = 1.0, mono_weight: float = 1.0):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids
        self.pos_token_id = reward_token_ids[0] if reward_token_ids is not None else None
        self.neg_token_id = reward_token_ids[1] if reward_token_ids is not None else None
        self.eps = 1e-5
        self.neg_weight = neg_weight
        self.mono_weight = mono_weight
        
    def forward(self, inputs: torch.Tensor, values: torch.Tensor, labels: torch.Tensor,  *, return_acc: bool = False):
        B, L = values.shape
       
        placeholder_mask = inputs == self.placeholder_token_id
        
        is_wrong = placeholder_mask & (labels == self.neg_token_id)
        
        indices = torch.arange(L, device=inputs.device).expand(B, L)

        last_placeholder_indices = (indices * placeholder_mask.long()).max(dim=1).values

        # 2. 提取最后一步的 label
        # 结果形状为 [Batch]
        last_step_labels = labels.gather(1, last_placeholder_indices.unsqueeze(1)).squeeze(1)

        # 3. 判定是否为负样本
        # 只有当样本包含占位符，且最后一个占位符对应的 label 是 neg_token 时，才算负样本
        has_placeholder = placeholder_mask.any(dim=1)
        is_negative_sample = has_placeholder & (last_step_labels == self.neg_token_id)
        # print(is_negative_sample)
        # 4. 判定是否为正样本
        is_positive_sample = has_placeholder & ~is_negative_sample
        # print(is_positive_sample)
        # print(is_negative_sample)
        # is_negative_sample = is_wrong.any(dim=1)
        # is_positive_sample = placeholder_mask.any(dim=1) & ~is_negative_sample
        

        log_h = F.logsigmoid(values)             
        log_1_m_h = F.logsigmoid(-values)           
        log_1_m_h_masked = log_1_m_h * placeholder_mask.float()
       
        # 公式 11: L_S = -sum(log(1 - h(t)))
        L_S_batch = -log_1_m_h_masked.sum(dim=1) 
        # print(L_S_batch)
        # quit()
        # 公式 12: L_W
        # S_T_batch = exp(sum(log(1 - h(t))))
        S_T_batch = torch.exp(log_1_m_h_masked.sum(dim=1))
        L_W_batch = -torch.log(torch.clamp(1.0 - S_T_batch, min=self.eps))

        # 公式 13: L_z
        z_indices = is_wrong.float().argmax(dim=1) 
        batch_idx = torch.arange(B, device=values.device)
        
        # 提取第一个出错位置的 log(h(z_i))
        log_h_zi = log_h[batch_idx, z_indices] 

        pos = torch.arange(L, device=values.device).unsqueeze(0).expand(B, L)
        before_zi_mask = placeholder_mask & (pos < z_indices.unsqueeze(1))
        
        # 累加 z_i 之前的 log(1 - h(t))
        sum_log_1_m_h_before_zi = (log_1_m_h * before_zi_mask.float()).sum(dim=1) 
        L_z_batch = -(log_h_zi + sum_log_1_m_h_before_zi) 

        #### 单调性约束
        h_probs = torch.sigmoid(values)
        h_t = h_probs[:, :-1]
        h_next = h_probs[:, 1:]
        diff = h_t - h_next  # h(t) - h(t+1)
        margin = 5e-2
        
        # A. 负样本单调性：h(t) 应该不减 (h_next >= h_t)，即 diff 应该 <= 0
        # 覆盖范围：从第一个错误 z_i 开始往后 (包括转折点 zi-1 到 zi)
        pair_mask_neg = placeholder_mask[:, :-1] & (indices[:, 1:] >= z_indices.unsqueeze(1))
        L_mono_neg = (torch.relu(diff + margin) * pair_mask_neg.float()).sum(dim=1)
        
        # B. 正样本单调性：h(t) 应该不增 (h_next <= h_t)，即 diff 应该 >= 0
        # 如果你坚持正样本错误率要越来越小，则惩罚 diff < 0 的情况
        pair_mask_pos = placeholder_mask[:, :-1] & placeholder_mask[:, 1:]
        L_mono_pos = (torch.relu(-diff + margin) * pair_mask_pos.float()).sum(dim=1)
        
        # L_mono_batch = (-torch.log(h_probs + 1e-8) * after_zi_mask.float()).sum(dim=1)
        ###########
        
        # 整合 Loss
        total_loss_batch = torch.zeros(B, device=values.device)
        total_loss_batch[is_positive_sample] = L_S_batch[is_positive_sample] + self.mono_weight * L_mono_pos[is_positive_sample]
        total_loss_batch[is_negative_sample] = self.neg_weight * (L_W_batch[is_negative_sample] + L_z_batch[is_negative_sample] + self.mono_weight * L_mono_neg[is_negative_sample])

        valid_samples_count = is_positive_sample.sum() + is_negative_sample.sum()
        loss = total_loss_batch.sum() / torch.clamp(valid_samples_count.float(), min=1.0)
        loss_neg = total_loss_batch[is_negative_sample].sum() / torch.clamp(is_negative_sample.sum().float(), min=1.0)
        loss_pos = total_loss_batch[is_positive_sample].sum() / torch.clamp(is_positive_sample.sum().float(), min=1.0)
        if return_acc:
            probs = torch.sigmoid(values)
    
            # 2. 提取掩码和标签
            is_neg_step = placeholder_mask & (labels == self.neg_token_id)
            is_pos_step = placeholder_mask & (labels == self.pos_token_id)
            
            # 3. 计算各部分的准确率张量
            # 负样本：prob 越高越对；正样本：(1-prob) 越高越对
            acc_neg_tensor = probs[is_neg_step]
            acc_pos_tensor = 1.0 - probs[is_pos_step]
            
            # 4. 计算各部分数量
            n_neg = is_neg_step.sum().float()
            n_pos = is_pos_step.sum().float()
            n_total = n_neg + n_pos
            
            # 5. 指标汇总
            # 总体准确率 (Total Acc) - 反应全局拟合情况
            acc = (acc_neg_tensor.sum() + acc_pos_tensor.sum()) / torch.clamp(n_total, min=1.0)
            
            # 拆分准确率 - 反应预测偏好
            acc_neg = acc_neg_tensor.sum() / torch.clamp(n_neg, min=1.0)
            acc_pos = acc_pos_tensor.sum() / torch.clamp(n_pos, min=1.0)
            
            # 6. 计算“平衡准确率” (Balanced Acc) 
            # 如果正负样本极度不平衡，这个指标比 total_acc 更能反映真实能力
            # balanced_acc = (acc_neg + acc_pos) / 2.0
            
        else:
            acc = torch.tensor(0.0, device=values.device)
            acc_neg = torch.tensor(0.0, device=values.device)
            acc_pos = torch.tensor(0.0, device=values.device)
        return loss,loss_pos,loss_neg, acc, acc_pos, acc_neg
