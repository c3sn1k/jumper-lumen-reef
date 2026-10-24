"""Temple stairs: a fixed staircase prop and a training-only look.

The staircase is a **prop**, not terrain: it is built by the `jumper.stairs` task
itself (the same way `jumper.swing` builds its swing), so the task cannot train on
an empty field when someone forgets `--scene`. This module owns the one geometry
description; the task and any replay scene use it rather than keeping a second
copy of the step heights.

Ten steps, 0.30 m run and 1.8 m wide. The rises go 0.03 -> 0.06 m and the total
top is 0.44 m. The first steps are lower and the last two are the hard ones; the
task's foot-lift target is raised to match.
"""

from __future__ import annotations

#: Local x where the first step starts, relative to the entity's placed origin.
STAIRS_X = 0.6
#: Ten rises, in metres, in climbing order.
RISES = (0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06)
RUN = 0.30
WIDTH = 1.8
#: Height of the last step's top, in metres.
TOP_HEIGHT = float(sum(RISES))


def _stairs_spec():
    """The staircase as an MjSpec: one fixed body with ten box geoms.

    No free joint, so mjlab wraps the body in a mocap frame and places one
    staircase per environment. The task installs the reset event that writes
    the placement; without it every staircase would stand at the world origin.
    """
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="temple_stairs", pos=[0.0, 0.0, 0.0])
    top = 0.0
    for index, rise in enumerate(RISES):
        top += rise
        body.add_geom(
            name=f"stair_{index:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[RUN / 2.0, WIDTH / 2.0, top / 2.0],
            pos=[index * RUN + RUN / 2.0, 0.0, top / 2.0],
        )
    return spec


def stairs_entity():
    """The staircase as an mjlab entity, placed relative to each environment."""
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=_stairs_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(STAIRS_X, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
    )


def scene():
    """A dark, cool training look. The steps themselves come from the task."""
    from .registry import Headlight, Scene

    return Scene(
        headlight=Headlight(ambient=(0.20, 0.22, 0.26), diffuse=(0.55, 0.58, 0.62)),
    )
