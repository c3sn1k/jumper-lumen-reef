#!/usr/bin/env python3
"""Build a pedestal warm-start checkpoint from a stairs actor.

The stairs and pedestal tasks share the tripod skeleton but not their widths:
stairs drives the 20 gait joints, the pedestal drives those plus the two J4
gripper joints. A direct ``--resume`` from the stairs checkpoint therefore fails
on the actor's first layer and action head.

This tool maps the shared gait weights into the pedestal observation layout,
initializes the new gripper columns and rows, and writes a checkpoint that
``scripts/train.py --resume`` accepts. Only the actor is transferred; the critic
and optimizer are fresh because the pedestal's observations, reward and value
landscape are different.

Usage (from the repository root):

    python tools/warmstart_pedestal.py \
        --source checkpoints/model_1000.pt \
        --out /content/session/pedestal_warmstart.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from tasks import load_agent_cfg
from tasks.jumper.common.constants import GAIT_JOINTS, GRIPPER_JOINTS

#: The actor observation history, matching `common.velocity_env.OBS_HISTORY`.
OBS_HISTORY = 5

#: A modest inherited sample count. A zero count lets a one-environment smoke
#: run set the variance from a single sample (std -> 0) and blow up the
#: normalised observation; the original 6M count would make the new gripper
#: dimensions adapt too slowly. 1024 is the compromise.
NORMALIZER_COUNT = 1024

#: Widths of the scalar observation terms around the per-joint blocks. The
#: order is: base_ang_vel, projected_gravity, joint_pos, joint_vel, actions,
#: command, actuator_force, gait_phase.
_SCALAR_HEAD = 3 + 3
_SCALAR_MIDDLE = 3
_SCALAR_TAIL = 2


def _joint_block_indices(
    old_joints: int,
    new_joints: int,
    history: int = OBS_HISTORY,
) -> torch.Tensor:
    """Map each new observation column to its old column, or -1 if new.

    The four per-joint blocks are joint_pos, joint_vel, actions and
    actuator_force; the command block sits between actions and actuator_force,
    so the blocks cannot simply be iterated as one run.
    """
    old_obs = _obs_width(old_joints, history)
    new_obs = _obs_width(new_joints, history)
    mapping = torch.full((new_obs,), -1, dtype=torch.long)

    def copy_scalar(old_start: int, new_start: int, width: int) -> None:
        mapping[new_start:new_start + width] = torch.arange(
            old_start, old_start + width
        )

    def copy_joint_block(old_start: int, new_start: int) -> None:
        for frame in range(history):
            old = old_start + frame * old_joints
            new = new_start + frame * new_joints
            mapping[new:new + old_joints] = torch.arange(
                old, old + old_joints
            )

    old = 0
    new = 0
    copy_scalar(old, new, _SCALAR_HEAD)
    old += _SCALAR_HEAD
    new += _SCALAR_HEAD

    for _ in ("joint_pos", "joint_vel", "actions"):
        copy_joint_block(old, new)
        old += history * old_joints
        new += history * new_joints

    copy_scalar(old, new, _SCALAR_MIDDLE)
    old += _SCALAR_MIDDLE
    new += _SCALAR_MIDDLE

    copy_joint_block(old, new)
    old += history * old_joints
    new += history * new_joints

    copy_scalar(old, new, _SCALAR_TAIL)
    old += _SCALAR_TAIL
    new += _SCALAR_TAIL

    if old != old_obs or new != new_obs:
        raise AssertionError(
            f"observation layout mismatch: walked {old}/{new}, "
            f"expected {old_obs}/{new_obs}"
        )
    return mapping


def _obs_width(joints: int, history: int = OBS_HISTORY) -> int:
    return _SCALAR_HEAD + 4 * history * joints + _SCALAR_MIDDLE + _SCALAR_TAIL


def _mapped_actor_state(
    old_state: dict[str, torch.Tensor],
    old_joints: int,
    new_joints: int,
) -> dict[str, torch.Tensor]:
    mapping = _joint_block_indices(old_joints, new_joints)
    new_obs = _obs_width(new_joints)
    mapped = mapping >= 0

    result: dict[str, torch.Tensor] = {}
    for key, value in old_state.items():
        if key == "mlp.0.weight":
            new_value = torch.zeros(
                value.shape[0], new_obs, dtype=value.dtype
            )
            new_value[:, mapped] = value[:, mapping[mapped]]
        elif key.startswith("obs_normalizer."):
            suffix = key.split(".", 1)[1]
            if suffix == "count":
                new_value = torch.full_like(value, NORMALIZER_COUNT)
            else:
                fill = 1.0 if suffix in ("_var", "_std") else 0.0
                new_value = torch.full(
                    (1, new_obs), fill, dtype=value.dtype
                )
                new_value[:, mapped] = value[:, mapping[mapped]]
        elif key == "distribution.std_param":
            new_value = torch.ones(new_joints, dtype=value.dtype)
            new_value[:old_joints] = value
        elif key == "mlp.8.weight":
            new_value = torch.zeros(
                new_joints, value.shape[1], dtype=value.dtype
            )
            new_value[:old_joints] = value
        elif key == "mlp.8.bias":
            new_value = torch.zeros(new_joints, dtype=value.dtype)
            new_value[:old_joints] = value
        else:
            new_value = value.clone()
        result[key] = new_value

    expected = _obs_width(new_joints)
    if result["mlp.0.weight"].shape[1] != expected:
        raise AssertionError("first layer was not mapped to the new width")
    return result


def _fresh_critic_state(
    template: dict[str, torch.Tensor],
    new_obs: int,
) -> dict[str, torch.Tensor]:
    """A correctly shaped but untrained critic.

    The template supplies the key set and hidden shapes, so a strict
    ``load_state_dict`` succeeds. The first layer is resized to the pedestal's
    critic width; all weights are re-initialized because the stairs value
    function describes a different task and reward.
    """
    result: dict[str, torch.Tensor] = {}
    for key, value in template.items():
        if key.startswith("obs_normalizer."):
            suffix = key.split(".", 1)[1]
            if suffix == "count":
                result[key] = torch.full_like(value, NORMALIZER_COUNT)
            else:
                fill = 1.0 if suffix in ("_var", "_std") else 0.0
                result[key] = torch.full(
                    (1, new_obs), fill, dtype=value.dtype
                )
        elif key == "mlp.0.weight":
            weight = torch.empty(
                value.shape[0], new_obs, dtype=value.dtype
            )
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            result[key] = weight
        elif key.endswith(".weight") and value.ndim == 2:
            weight = torch.empty_like(value)
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            result[key] = weight
        elif key.endswith(".bias"):
            result[key] = torch.zeros_like(value)
        else:
            result[key] = value.clone()
    return result


def _fresh_optimizer_state(lr: float, parameter_count: int) -> dict:
    """An empty Adam state with the same parameter-group arity.

    ``load_state_dict`` checks the number of tensors in the group even when the
    per-parameter state is empty, so the dummy optimizer has to expose as many
    parameters as the real actor+critic.
    """
    parameters = [
        torch.nn.Parameter(torch.zeros(1))
        for _ in range(parameter_count)
    ]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    return optimizer.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "checkpoints"
        / "model_1000.pt",
        help="stairs checkpoint to read (default: checkpoints/model_1000.pt)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write the pedestal warm-start checkpoint",
    )
    parser.add_argument(
        "--task",
        default="jumper.pedestal",
        help="pedestal task id, used for the learning rate",
    )
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"error: no such checkpoint: {args.source}")

    checkpoint = torch.load(
        args.source, map_location="cpu", weights_only=False
    )
    old_actor = checkpoint["actor_state_dict"]
    old_critic = checkpoint["critic_state_dict"]

    old_joints = len(GAIT_JOINTS)
    new_joints = old_joints + len(GRIPPER_JOINTS)
    old_obs = old_actor["mlp.0.weight"].shape[1]
    if old_obs != _obs_width(old_joints):
        raise SystemExit(
            f"error: {args.source} has actor obs {old_obs}, expected "
            f"{_obs_width(old_joints)} for {old_joints} gait joints"
        )
    if old_actor["mlp.8.weight"].shape[0] != old_joints:
        raise SystemExit(
            f"error: {args.source} has {old_actor['mlp.8.weight'].shape[0]} "
            f"action rows, expected {old_joints}"
        )

    new_obs = _obs_width(new_joints)
    critic_obs = old_critic["mlp.0.weight"].shape[1] + (
        new_obs - old_obs
    )
    agent_cfg = load_agent_cfg(args.task)
    learning_rate = float(agent_cfg.algorithm.learning_rate)
    parameter_count = len(
        checkpoint["optimizer_state_dict"]["param_groups"][0]["params"]
    )

    warm = {
        "actor_state_dict": _mapped_actor_state(
            old_actor, old_joints, new_joints
        ),
        "critic_state_dict": _fresh_critic_state(old_critic, critic_obs),
        "optimizer_state_dict": _fresh_optimizer_state(
            learning_rate, parameter_count
        ),
        "iter": 0,
        "infos": {},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(warm, args.out)
    print(
        f"warm start written: {args.out}\n"
        f"  actor obs: {old_obs} -> {new_obs} "
        f"({new_joints - old_joints} gripper joints)\n"
        f"  action: {old_joints} -> {new_joints}\n"
        f"  critic: fresh, {critic_obs} obs\n"
        f"  optimizer: fresh Adam, lr={learning_rate:g}\n"
        f"  iteration: 0"
    )


if __name__ == "__main__":
    main()
