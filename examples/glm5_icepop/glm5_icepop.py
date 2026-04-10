"""GLM-5 style improved IcePop off-policy correction.

Unlike standard IcePop which computes the importance ratio as
π_θ_old / π_rollout (requiring an old-actor forward pass), GLM-5 drops
π_θ_old entirely and uses r_t(θ) = π_θ / π_rollout directly.

This avoids maintaining stale policy checkpoints in asynchronous training
while keeping IcePop's asymmetric clipping + out-of-range token masking.

Usage:
    --use-rollout-logprobs \
    --use-tis \
    --custom-tis-function-path examples/glm5_icepop/glm5_icepop.py:glm5_icepop_function \
    --tis-clip <upper_bound> \
    --tis-clip-low <lower_bound> \
    --eps-clip <ppo_lower_clip> \
    --eps-clip-high <ppo_upper_clip> \
    --reset-optimizer-states
"""

from typing import Any

import torch


def glm5_icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    current_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """GLM-5 IcePop: use r_t(θ) = π_θ / π_rollout directly, drop π_θ_old.

    Applies asymmetric masking: tokens where the ratio falls outside
    [tis_clip_low, tis_clip] are zeroed out entirely (not clamped).

    Args:
        args: Configuration with tis_clip_low and tis_clip attributes.
        pg_loss: Policy gradient loss tensor (1D, concatenated across sequences).
        current_log_probs: Log probs from current training forward pass. list of 1D tensors.
        rollout_log_probs: Log probs recorded during rollout (inference engine). list of 1D tensors.
        loss_masks: Response masks. list of 1D tensors.
        **kwargs: Absorbs extra arguments (train_log_probs, etc.) for interface compatibility.

    Returns:
        Tuple of (modified pg_loss, loss_masks, metrics dict).
    """
    assert current_log_probs is not None, (
        "current_log_probs is None. GLM-5 IcePop requires --use-rollout-logprobs to be set."
    )

    rollout_lp = torch.cat(rollout_log_probs, dim=0)
    current_lp = torch.cat(current_log_probs, dim=0)

    # r_t(θ) = π_θ / π_rollout
    ice_ratio = torch.exp(current_lp - rollout_lp)
    ice_abs = (ice_ratio - 1).abs()

    # IcePop-style masking: zero out tokens where ratio is outside [low, high]
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip),
        ice_ratio,
        torch.zeros_like(ice_ratio),
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()

    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }

    # Detach to prevent gradients flowing through the masking ratio,
    # consistent with original IcePop where old_log_probs is detached.
    pg_loss = pg_loss * ice_weight.detach()
    return pg_loss, loss_masks, metrics
