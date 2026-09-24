"""Local Go2 Walk These Ways controller with simulated pose feedback.

Uses the pinned go2_deploy policy and its observation/action conventions.
This module has no DDS, SDK2, or physical robot communication code.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

from policy_assets import POLICY_DIR, check_assets

JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
               for joint in ("hip", "thigh", "calf")]
GAIT = (0.5, 0.30, 0.08, 0.0)  # period, body height, foot clearance, pitch


def projected_gravity(quat):
    w, x, y, z = quat
    return np.array([2*(w*y-x*z), -2*(w*x+y*z), 2*(x*x+y*y)-1], dtype=np.float32)


class WalkingPolicy:
    """Five 61-value frames, oldest first, as in go2_deploy's WTWController."""

    def __init__(self):
        check_assets()
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(POLICY_DIR / "wtw_model.pt"), map_location="cpu").eval()
        self.default = np.tile([0.0, 0.8, -1.5], 4).astype(np.float32)
        self.reset()

    def reset(self):
        self.action = np.zeros(12, dtype=np.float32)
        self.history = deque([np.zeros(61, dtype=np.float32) for _ in range(5)], maxlen=5)
        self.gait_time = 0.0

    def step(self, q, dq, quat, gyro, command):
        period, height, clearance, pitch = GAIT
        self.gait_time += 0.02
        if self.gait_time > period - 0.01:
            self.gait_time = 0.0
        theta = np.array([0.0, 0.5, 0.5, 0.0])  # trot, FL/FR/RL/RR phase offsets
        phase = 2*np.pi*(self.gait_time/period + theta)
        obs = np.concatenate([
            np.asarray(command) * [1, 1, .25], projected_gravity(quat), gyro*.25,
            q-self.default, dq*.05, self.action, np.sin(phase), np.cos(phase),
            [period, height, clearance, pitch], theta,
        ]).astype(np.float32)
        self.history.append(obs)
        inputs = np.concatenate(self.history)[None]
        with torch.inference_mode():
            self.action = self.model(torch.from_numpy(inputs)).numpy().reshape(12).copy()
        if not np.isfinite(self.action).all():
            raise RuntimeError("Policy returned a non-finite action")
        return self.default + self.action * .25


class SimSportClient:
    """Local Move/StopMove subset, with real MuJoCo physics and no SDK connection.

    Move takes body-frame m/s and rad/s. The reference trajectory uses the
    requested heading and velocity; feedback uses MuJoCo's actual base pose.
    Commands persist until replaced. step() advances 20 ms of simulated time.
    """

    period = 0.02

    def __init__(self, scene: Path, feedback: bool = True):
        self.policy = WalkingPolicy()
        self.feedback = feedback
        spec = mujoco.MjSpec.from_file(str(scene))
        if scene.name == "scene.xml":
            # Unitree's default scene contains hurdles. The walking default is
            # an unobstructed plane; --terrain retains its actual obstacles.
            for geom in list(spec.geoms):
                if geom.parent.name == "world" and geom.name != "floor":
                    spec.delete(geom)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.substeps = round(self.period / self.model.opt.timestep)
        if not np.isclose(self.substeps*self.model.opt.timestep, self.period):
            raise ValueError("Policy period must be a multiple of the physics timestep")
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        if any(i < 0 or int(self.model.jnt_type[i]) != int(mujoco.mjtJoint.mjJNT_HINGE) for i in ids):
            raise ValueError("Expected the twelve named Go2 hinge joints")
        self.qadr, self.dadr = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.ranges = self.model.jnt_range[ids]
        free = np.flatnonzero(self.model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
        if len(free) != 1:
            raise ValueError("Expected one floating Go2 base")
        self.base_q = int(self.model.jnt_qposadr[free[0]])
        self.base_d = int(self.model.jnt_dofadr[free[0]])
        self.default = self.policy.default
        self.reset()

    def reset(self, yaw: float = 0):
        mujoco.mj_resetData(self.model, self.data)
        qa = self.base_q
        self.data.qpos[qa:qa+7] = (-2, 0, .35, np.cos(yaw/2), 0, 0, np.sin(yaw/2))
        self.data.qpos[self.qadr] = self.default
        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):
            self._advance(self.default, 60, 5)
        self.policy.reset()
        self.StopMove()
        self.was_moving = False
        self.stop_pose = self.default.copy()
        self.stop_elapsed = .5
        self.inference_ms = 0.0
        self._reset_reference()
        self.check_upright()

    def _reset_reference(self):
        self.reference_xy = self.data.qpos[self.base_q:self.base_q+2].copy()
        self.reference_yaw = self.metrics()["yaw"]
        self.yaw_integral = 0.0

    def Move(self, vx: float, vy: float, yaw_rate: float):
        values = np.asarray([vx, vy, yaw_rate], dtype=float)
        if not np.isfinite(values).all() or (np.abs(values) > [.6, .3, .5]).any():
            raise ValueError("Demo limits: finite |vx| <= 0.6, |vy| <= 0.3, |yaw_rate| <= 0.5")
        self.command = tuple(float(v) for v in values)

    def StopMove(self):
        self.command = (0.0, 0.0, 0.0)

    def _corrected_command(self):
        if not self.feedback:
            return self.command
        m = self.metrics()
        self.reference_yaw += self.command[2]*self.period
        c, s = np.cos(self.reference_yaw), np.sin(self.reference_yaw)
        world_velocity = np.array([[c, -s], [s, c]]) @ np.array(self.command[:2])
        self.reference_xy += world_velocity*self.period
        error = self.reference_xy-self.data.qpos[self.base_q:self.base_q+2]
        if np.linalg.norm(error) > .5:
            raise RuntimeError("Robot is over 0.5 m behind its target; reset before continuing")
        c, s = np.cos(m["yaw"]), np.sin(m["yaw"])
        corrected_xy = np.array([[c, s], [-s, c]]) @ (world_velocity+1.2*error)
        yaw_error = np.arctan2(np.sin(self.reference_yaw-m["yaw"]), np.cos(self.reference_yaw-m["yaw"]))
        self.yaw_integral = float(np.clip(self.yaw_integral+yaw_error*self.period, -.3, .3))
        corrected_yaw = self.command[2]+2*yaw_error+.5*self.yaw_integral-.1*(m["yaw_rate"]-self.command[2])
        return (*np.clip(corrected_xy, -.8, .8), float(np.clip(corrected_yaw, -1, 1)))

    def _advance(self, target, kp, kd):
        target = np.clip(target, self.ranges[:, 0], self.ranges[:, 1])
        for _ in range(self.substeps):
            torque = kp*(target-self.data.qpos[self.qadr])-kd*self.data.qvel[self.dadr]
            self.data.ctrl[:] = 0
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.dadr] = np.clip(torque, -23.5, 23.5)
            mujoco.mj_step(self.model, self.data)

    def step(self):
        q = self.data.qpos[self.qadr].astype(np.float32)
        if self.command == (0.0, 0.0, 0.0):
            # Smoothly stop the gait and hold using PD; physics keeps running.
            if self.was_moving:
                self.stop_pose, self.stop_elapsed = q.copy(), 0.0
            self.was_moving = False
            self.stop_elapsed += self.period
            alpha = min(1, self.stop_elapsed/.5)
            self._advance((1-alpha)*self.stop_pose+alpha*self.default,
                          (1-alpha)*20+alpha*60, (1-alpha)*.5+alpha*5)
            self.inference_ms = 0.0
        else:
            if not self.was_moving:
                self.policy.reset()
                self._reset_reference()
            self.was_moving = True
            command = self._corrected_command()
            started = time.perf_counter()
            target = self.policy.step(
                q, self.data.qvel[self.dadr].astype(np.float32),
                self.data.qpos[self.base_q+3:self.base_q+7].astype(np.float32),
                self.data.qvel[self.base_d+3:self.base_d+6].astype(np.float32), command,
            )
            self.inference_ms = (time.perf_counter()-started)*1000
            self._advance(target, 20, .5)
        self.check_upright()

    def metrics(self):
        qa, da = self.base_q, self.base_d
        w, x, y, z = self.data.qpos[qa+3:qa+7]
        return dict(x=float(self.data.qpos[qa]), y=float(self.data.qpos[qa+1]),
                    height=float(self.data.qpos[qa+2]),
                    yaw=float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                    tilt_deg=float(np.degrees(np.arccos(np.clip(1-2*(x*x+y*y), -1, 1)))),
                    speed=float(np.linalg.norm(self.data.qvel[da:da+2])),
                    yaw_rate=float(self.data.qvel[da+5]))

    def check_upright(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Simulation produced non-finite state")
        m = self.metrics()
        if m["height"] < .15 or m["tilt_deg"] > 45:
            raise RuntimeError(f"Go2 fell: height={m['height']:.3f} m, tilt={m['tilt_deg']:.1f} deg")


DEMO = [(2, "stand", (0,0,0)), (10, "straight forward", (.4,0,0)),
        (2, "stop", (0,0,0)), (4, "left", (0,.3,0)), (4, "right", (0,-.3,0)),
        (4, "turn left", (0,0,.5)), (4, "turn right", (0,0,-.5)),
        (4, "backward", (-.3,0,0)), (3, "stop", (0,0,0))]


def demo_command(t):
    for duration, label, command in DEMO:
        if t < duration:
            return label, command
        t -= duration
    return "stop", (0,0,0)


def run_viewer(client, demo, auto_close_seconds):
    import mujoco.viewer

    keys = {ord("W"): (.4,0,0), ord("S"): (-.3,0,0), ord("A"): (0,.3,0),
            ord("D"): (0,-.3,0), ord("Q"): (0,0,.5), ord("E"): (0,0,-.5), 32: (0,0,0)}
    pending = deque()

    def on_key(key):
        pending.append(key)

    print("Walk These Ways + simulated pose feedback. W/S: forward/back; A/D: sideways;")
    print("Q/E: turn; Space: stop and stand; R: reset. Tap sets a persistent command.")
    if demo:
        print("37-second demo. A movement key switches to manual control.")
    started, sim_start, phase = time.monotonic(), client.data.time, None
    with mujoco.viewer.launch_passive(client.model, client.data, key_callback=on_key) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135, -25
        while viewer.is_running():
            tick = time.monotonic()
            while pending:
                key = pending.popleft()
                if key in keys:
                    demo = False
                    client.Move(*keys[key])
                    print(f"Command: {client.command}", flush=True)
                elif key == ord("R"):
                    demo = False
                    with viewer.lock():
                        client.reset()
                    print("Reset to the starting pose.", flush=True)
            if demo:
                label, command = demo_command(client.data.time-sim_start)
                client.Move(*command)
                if label != phase:
                    print(f"Demo: {label}", flush=True)
                    phase = label
            with viewer.lock():
                client.step()
                viewer.cam.lookat[:] = client.data.qpos[client.base_q:client.base_q+3]
            viewer.sync()
            if auto_close_seconds and time.monotonic()-started >= auto_close_seconds:
                break
            time.sleep(max(0, client.period-(time.monotonic()-tick)))
    print(f"Viewer closed. Final position: {client.metrics()}")
    return 0
