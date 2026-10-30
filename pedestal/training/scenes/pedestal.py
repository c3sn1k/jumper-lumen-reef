"""The pedestal apparatus for `jumper.pedestal`.

The final Lumen Temple has a platform at 0.44 m and a round pedestal on it whose
top is another 0.24 m above that. The training apparatus here is the same shape:
a wide platform the robot starts on, and a tall cylinder placed so only the
requested height is exposed above the platform.

The cylinder is **taller than the exposed step on purpose**. Its body is placed
lower than the platform top, so the part below the platform is hidden inside it.
Raising or lowering the body therefore changes the exposed step without changing
the collision geometry, which is what lets one fixed MJCF serve a height
curriculum. The pedestal and platform are fixed-base props; mjlab wraps them in
mocap frames and the task's reset events place them per environment.
"""

from __future__ import annotations

#: Top of the platform the robot starts on, in metres.
PLATFORM_TOP = 0.44
#: Half extents of the platform, in metres: 2.4 m long, 1.8 m wide, top at 0.44.
PLATFORM_HALF_SIZE = (1.2, 0.9, 0.22)

#: The round pedestal's radius, in metres -- the final pedestal's 0.38.
PEDESTAL_RADIUS = 0.38
#: Half the cylinder's **body** height. The body is 0.40 m tall; the exposed
#: step is the part above the platform, from 0.08 to 0.24 m.
PEDESTAL_HALF_HEIGHT = 0.20
#: Where the pedestal stands on the platform.
PEDESTAL_X = 0.6
PEDESTAL_Y = 0.0

#: Exposed step range above the platform, in metres.
HEIGHT_MIN = 0.08
HEIGHT_MAX = 0.24
#: Body centre height at `HEIGHT_MIN`. The reset event adds the curriculum's
#: extra height on top of this.
PEDESTAL_BASE_Z = PLATFORM_TOP + HEIGHT_MIN - PEDESTAL_HALF_HEIGHT


def _platform_spec():
    """The platform as an MjSpec: one fixed body with a single box geom."""
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="pedestal_platform", pos=[0.0, 0.0, 0.0])
    body.add_geom(
        name="pedestal_platform",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(PLATFORM_HALF_SIZE),
        pos=[0.0, 0.0, PLATFORM_HALF_SIZE[2]],
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.18, 0.21, 0.24, 1.0],
    )
    return spec


def _pedestal_spec():
    """The pedestal as an MjSpec: one fixed body with a single cylinder geom."""
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="pedestal", pos=[0.0, 0.0, 0.0])
    body.add_geom(
        name="pedestal",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[PEDESTAL_RADIUS, PEDESTAL_HALF_HEIGHT],
        pos=[0.0, 0.0, 0.0],
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.20, 0.24, 0.28, 1.0],
    )
    return spec


def platform_entity():
    """The platform as an mjlab entity, placed at each environment's origin."""
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=_platform_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
    )


def pedestal_entity():
    """The pedestal as an mjlab entity, at the lowest curriculum height."""
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=_pedestal_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(PEDESTAL_X, PEDESTAL_Y, PEDESTAL_BASE_Z),
            joint_pos={".*": 0.0},
        ),
    )


def pedestal_top(*, height: float | None = None) -> float:
    """World z of the exposed pedestal top at `height` above the platform."""
    if height is None:
        height = HEIGHT_MIN
    return PLATFORM_TOP + height
