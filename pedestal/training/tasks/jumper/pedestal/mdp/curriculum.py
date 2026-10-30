"""A height curriculum for the pedestal.

A class term, like `CommandRangeCurriculum`, because the level has to survive
between calls. The pedestal body is a fixed geometry placed lower than the
platform; raising its reset z raises only the exposed step, so the curriculum
changes the task without rebuilding the model.

The schedule is deliberately simple for the first version: one rung every
`dwell_steps` control steps. `common_step_counter` is per-environment control
steps (it was 52,824 at iteration 2,199 with 24 steps per iteration in the
staircase run), so 4,000 steps is about 167 PPO iterations per rung. It is a
schedule rather than a performance gate because there is no measured mount rate
yet to gate on; once a run exists, the gate should replace the counter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.curriculum_manager import CurriculumTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class PedestalHeightCurriculum:
    """Raise the exposed pedestal step from `levels[0]` to `levels[-1]`."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        params = cfg.params or {}
        self.levels = tuple(float(v) for v in params["levels"])
        self.height_min = float(params["height_min"])
        self.dwell_steps = int(params["dwell_steps"])
        self.event_name = str(params.get("event_name", "reset_prop_pedestal"))
        self.level = 0
        self._applied = False

    def _apply(self, env: ManagerBasedRlEnv) -> None:
        height = self.levels[self.level]
        span = max(0.0, height - self.height_min)
        term = env.event_manager.get_term_cfg(self.event_name)
        pose_range = term.params.setdefault("pose_range", {})
        # The pedestal's default z is the *lowest* rung. The reset offset is the
        # part above it; the body keeps the same geometry and only its exposed
        # height changes.
        pose_range["z"] = (0.0, span)
        self._applied = True

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids,
        levels: tuple[float, ...] | list[float] | None = None,
        height_min: float | None = None,
        dwell_steps: int | None = None,
        event_name: str | None = None,
    ) -> dict[str, float]:
        if levels is not None:
            self.levels = tuple(float(v) for v in levels)
        if height_min is not None:
            self.height_min = float(height_min)
        if dwell_steps is not None:
            self.dwell_steps = int(dwell_steps)
        if event_name is not None:
            self.event_name = str(event_name)

        step = int(env.common_step_counter)
        level = min(len(self.levels) - 1, step // max(1, self.dwell_steps))
        if level != self.level or not self._applied:
            self.level = level
            self._apply(env)
        return {"level": self.level, "height": self.levels[self.level]}
