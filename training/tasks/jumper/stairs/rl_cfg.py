"""Hyper-parameters for jumper.stairs.

The gait is tripod's, so the PPO baseline and runner are tripod's too. Only the
experiment name differs: the log path is `logs/<model>/jumper.stairs/<date-time>`.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..tripod.rl_cfg import (
    agent_cfg as _tripod_agent_cfg,
    play_status,
    runner_cls,
)


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = _tripod_agent_cfg()
    cfg.experiment_name = "jumper.stairs"
    return cfg
