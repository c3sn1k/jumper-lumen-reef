"""Hyper-parameters for jumper.pedestal.

The walking skeleton is tripod's, so the PPO baseline is tripod's numbers with
one deliberate exception: symmetry is **off**. The action and observation now
include the two grippers, and `tasks/jumper/common/mdp/symmetry.py` mirrors the
20 gait joints and their named observation terms; it does not describe the two
J4 joints or the extra widths they add. Turning the existing transform on would
either raise at startup or silently mirror a layout it does not understand. The
robot is left-right symmetric, but this task's first version is not using that
fact until the mirror map is extended.

The runner and play status are the tripod ones: the command is still a velocity
command, and the curriculum runner is what carries the pedestal height level in
the checkpoint.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline
from ..tripod.rl_cfg import play_status, runner_cls


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config; the log path is `logs/<model>/jumper.pedestal/`."""
    return jumper_ppo_baseline(
        experiment_name="jumper.pedestal",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=10_000,
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
    )
