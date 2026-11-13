"""Measure the robot's collision envelope during exploration motions in empty space.

This inspects only the robot's own collision geometry, never a house layout.
The finite motion sample is calibration evidence, not a formal envelope proof.
"""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'simulation')]

import mujoco
import numpy as np
from robot import SimGo2


def measure_envelope(sim, geoms, base):
    matrix = sim.data.xmat[base].reshape(3, 3)
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    co, sn = math.cos(yaw), math.sin(yaw)
    level = np.array([[co, sn, 0], [-sn, co, 0], [0, 0, 1]])
    maximum = np.zeros(2)
    for g in geoms:
        pos = level @ (sim.data.geom_xpos[g] - sim.data.xpos[base])
        rotation = level @ sim.data.geom_xmat[g].reshape(3, 3)
        kind, size = int(sim.model.geom_type[g]), sim.model.geom_size[g]
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            extent = np.full(3, size[0])
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            extent = abs(rotation[:, 2]) * size[1] + size[0]
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = abs(rotation) @ size
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            axis = rotation[:, 2]
            extent = abs(axis)*size[1] + np.sqrt(np.maximum(0., 1-axis*axis))*size[0]
        else:
            raise ValueError(f'unsupported collision primitive: {kind}')
        maximum = np.maximum(maximum, (abs(pos) + extent)[:2])
    return maximum


def calibrate():
    sim = SimGo2('empty')
    base = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    geoms = np.flatnonzero(sim._robot_geoms & ((sim.model.geom_contype != 0) |
                                               (sim.model.geom_conaffinity != 0)))
    motions = [('standing', (0., 0., 0.)), ('forward', (.15, 0., 0.)),
               ('backward', (-.15, 0., 0.)), ('left', (0., .15, 0.)),
               ('right', (0., -.15, 0.)), ('diagonal', (.106, .106, 0.)),
               ('turn_left', (0., 0., 1.)), ('turn_right', (0., 0., -1.))]
    results = []
    for name, command in motions:
        maximum = np.zeros(2)
        sim.Move(*command)
        for _ in range(200):
            sim.step()
            maximum = np.maximum(maximum, measure_envelope(sim, geoms, base))
        sim.StopMove()
        for _ in range(100):
            sim.step()
            maximum = np.maximum(maximum, measure_envelope(sim, geoms, base))
        result = dict(motion=name, half_length_m=float(maximum[0]),
                      half_width_m=float(maximum[1]), fault=sim.fault)
        results.append(result)
        print(json.dumps(result), flush=True)
    maximum = np.zeros(2)
    for command in [(.15, 0., 0.), (0., .15, 0.), (-.15, 0., 0.), (0., -.15, 0.)]*4:
        sim.Move(*command)
        for _ in range(8):
            sim.step()
            maximum = np.maximum(maximum, measure_envelope(sim, geoms, base))
        sim.StopMove()
        for _ in range(100):
            sim.step()
            maximum = np.maximum(maximum, measure_envelope(sim, geoms, base))
            state = sim.metrics()
            if math.hypot(state['vx'], state['vy']) < .05:
                break
    results.append(dict(motion='short_steps_and_braking', half_length_m=float(maximum[0]),
                        half_width_m=float(maximum[1]), fault=sim.fault))
    report = dict(frame='level body axes; yaw removed, roll/pitch retained',
                  units='metres', mujoco=mujoco.__version__, results=results)
    report['maximum_half_length_m'] = max(r['half_length_m'] for r in results)
    report['maximum_half_width_m'] = max(r['half_width_m'] for r in results)
    return report


if __name__ == '__main__':
    report = calibrate()
    if len(sys.argv)>1:
        output=Path(sys.argv[1])
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)
