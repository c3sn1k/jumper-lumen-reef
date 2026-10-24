"""Stairs-only reward terms."""

from __future__ import annotations

import torch


def climb_height(
    env,
    asset_cfg,
    start_height: float = 0.10,
    target_height: float = 0.35,
) -> torch.Tensor:
    """Normalised progress from the standing root height to the staircase top.

    Zero at the flat start and one once the root has risen by
    ``target_height``. The term is deliberately per-step rather than a sparse
    bonus at the top: a sparse signal gives PPO nothing to follow on the first
    runs, and the stairs are the only structure that can produce the height.
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    return torch.clamp((height - start_height) / max(target_height, 1e-6), 0.0, 1.0)


def backward_lean(
    env,
    asset_cfg,
    tilt_deadband: float = 0.35,
    backward_deadband: float = 0.05,
) -> torch.Tensor:
    """Penalty for a body pitched/rolled too far, or driven backwards.

    The term is larger the worse it is; the config carries the negative weight.
    ``projected_gravity_b``'s horizontal part is zero for a level body and grows
    as the body tilts; a backwards base velocity is penalised twice as strongly
    because that is the direction a failed mount falls.
    """
    asset = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    tilt = torch.clamp(
        torch.linalg.vector_norm(gravity[:, :2], dim=1) - tilt_deadband,
        min=0.0,
    )
    backward = torch.clamp(
        -asset.data.root_link_lin_vel_b[:, 0] - backward_deadband,
        min=0.0,
    )
    return tilt.square() + 2.0 * backward.square()
