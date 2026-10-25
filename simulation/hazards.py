"""The "hazards" scene (#22): a raised platform with a sheer ledge, stairs going down and a glass pane.

Everything the executor has to protect against without the model's help:
  - the east edge of the platform is a 0.5 m drop to the floor;
  - on the north side, four 12.5 cm steps lead down to the floor;
  - a glass pane stands on the platform's south half. It collides like any wall but sits in geom group 4,
    which the simulated LiDAR (and the viewer) don't see, so only the stall check can catch it.

Layout, world frame (m): platform x -3.5..1.0, y -2.0..2.0, top at z 0.5; the robot starts at (-2.5, 0) facing
east. The pane runs across y -2.0..-0.6 at x -1.0. Kept apart from world.py so the other scenes never change."""
from __future__ import annotations

import mujoco

PLATFORM = dict(x0=-3.5, x1=1.0, y0=-2.0, y1=2.0, top=.5)
STEP_RISE, STEP_RUN, STEPS = .125, .3, 4          # stairs down from the north edge
GLASS = dict(x=-1.0, y0=-2.0, y1=-.6, thick=.02, height=1.0)
START = (-2.5, 0.0, 0.0)

CONCRETE = (.7, .7, .68, 1)
STEP = (.62, .6, .56, 1)
GLASS_RGBA = (.7, .85, .95, .25)


def spec():
    """The world description SimGo2 uses for its start pose."""
    return {"start": list(START)}


def _box(spec_, name, centre, half, rgba, group=0):
    body = spec_.worldbody.add_body()
    body.name, body.pos = name, centre
    g = body.add_geom()
    g.type, g.size, g.rgba, g.group = mujoco.mjtGeom.mjGEOM_BOX, half, rgba, group
    g.friction = (.8, .02, .01)
    return body


def build_hazards(spec_, objects, world_object):
    p = PLATFORM
    _box(spec_, "platform", ((p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2, p["top"] / 2),
         ((p["x1"] - p["x0"]) / 2, (p["y1"] - p["y0"]) / 2, p["top"] / 2), CONCRETE)
    objects.append(("platform", world_object("platform", "platform", "grey")))
    # Stairs: a flight 1.2 m wide along x -2.6..-1.4, stepping down northwards from the platform edge.
    for i in range(STEPS):
        top = p["top"] - STEP_RISE * (i + 1)
        if top <= 0:
            break
        y0 = p["y1"] + STEP_RUN * i
        _box(spec_, f"step_{i}", (-2.0, y0 + STEP_RUN / 2, top / 2), (.6, STEP_RUN / 2, top / 2), STEP)
    g = GLASS
    _box(spec_, "glass_pane", (g["x"], (g["y0"] + g["y1"]) / 2, p["top"] + g["height"] / 2),
         (g["thick"] / 2, (g["y1"] - g["y0"]) / 2, g["height"] / 2), GLASS_RGBA, group=4)
    # Not listed in objects: the robot is not supposed to know it is there.
    spec_.visual.headlight.ambient = (.45, .45, .45)
