"""Compare pretrained policies using identical, unassisted MuJoCo trials.

Research harness: no DDS, no physical robot, no heading or velocity feedback.
Exact source URLs, revisions and hashes are in experiments/sources.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "_deps" / "policy-trials"
SCENE = ROOT / "_deps/unitree_mujoco/unitree_robots/go2/scene.xml"
JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
               for joint in ("hip", "thigh", "calf")]
CANDIDATES = ("himloco", "simple", "ts", "wtw", "ee", "waq")


def gravity(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([2 * (w*y - x*z), -2 * (w*x + y*z), 2*(x*x + y*y) - 1], dtype=np.float32)


class Candidate:
    def __init__(self, name: str, gait=(0.6, 0.34, 0.12, 0.3)):
        self.name = name
        self.gait = gait
        if name == "himloco":
            folder = TRIALS / "rl_sar/policy/go2/himloco"
            cfg = yaml.safe_load((folder / "config.yaml").read_text())["go2/himloco"]
            model_path = folder / cfg["model_name"]
            self.mapping = cfg["joint_mapping"]
            self.default = np.array(cfg["default_dof_pos"], dtype=np.float32)
            self.kp, self.kd = np.array(cfg["rl_kp"]), np.array(cfg["rl_kd"])
            self.scale = np.array(cfg["action_scale"], dtype=np.float32)
            self.torque_limit = np.array(cfg["torque_limits"])
            self.history_length = 6
            self.cmd_scale = np.array(cfg["commands_scale"])
        elif name in ("ee", "waq"):
            folder = TRIALS / "go2_deploy_python"
            cfg = yaml.safe_load((folder / "configs" / f"{name}.yaml").read_text())
            model_path = folder / cfg["policy_path"]
            self.mapping = cfg["leg_joint2motor_idx"]
            self.default = np.array(cfg["default_angles"], dtype=np.float32)
            self.kp, self.kd = cfg["ctrl_kp"], cfg["ctrl_kd"]
            self.scale = cfg["action_scale"]
            self.torque_limit = 23.5
            self.history_length = cfg["frame_stack"]
            self.cmd_scale = np.array(cfg["cmd_scale"])
        else:
            folder = TRIALS / "go2_deploy"
            config_name = "simple_rl" if name == "simple" else name
            cfg = yaml.safe_load((folder / "params" / f"{config_name}_config.yaml").read_text())
            model_path = folder / "models" / cfg["policy_name"]
            self.mapping = list(range(12))
            self.default = np.array(cfg["stand_pos"], dtype=np.float32)
            self.kp, self.kd = cfg["ctrl_kp"], cfg["ctrl_kd"]
            self.scale = cfg["action_scale"]
            self.torque_limit = 23.5
            self.history_length = cfg.get("frame_stack", 1)
            self.cmd_scale = np.array([cfg["lin_vel_scale"], cfg["lin_vel_scale"], cfg["ang_vel_scale"]])
        self.model_path = model_path
        self.module = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.reset()

    def reset(self):
        self.action = np.zeros(12, dtype=np.float32)
        self.dim = 61 if self.name == "wtw" else 45
        self.history = deque([np.zeros(self.dim, dtype=np.float32)
                              for _ in range(self.history_length)], maxlen=self.history_length)
        self.last_obs = np.zeros(self.dim, dtype=np.float32)
        self.gait_time = 0.0

    def inputs(self, q, dq, quat, gyro, command):
        q_rel, dq_scaled = q - self.default, dq * 0.05
        cmd = np.asarray(command) * self.cmd_scale
        if self.name == "himloco":
            obs = np.concatenate([cmd, gyro * 0.25, gravity(quat), q_rel, dq_scaled, self.action])
        else:
            obs = np.concatenate([cmd, gravity(quat), gyro * 0.25, q_rel, dq_scaled, self.action])
        if self.name == "wtw":
            # Match the C++ reset's first gait and upper-bound behaviour targets.
            period, height, clearance, pitch = self.gait
            theta = np.array([0.0, 0.5, 0.5, 0.0])
            self.gait_time += 0.02
            if self.gait_time > period - 0.01:
                self.gait_time = 0.0
            phase = 2 * np.pi * (self.gait_time / period + theta)
            obs = np.concatenate([obs, np.sin(phase), np.cos(phase),
                                  [period, height, clearance, pitch], theta])
        obs = obs.astype(np.float32)
        if self.name == "himloco":
            obs = np.clip(obs, -100, 100)
            self.history.appendleft(obs.copy())
        elif self.name == "ts":
            # The C++ exporter expects the previous twenty frames, excluding now.
            self.history.append(self.last_obs.copy())
        else:
            self.history.append(obs.copy())
        self.last_obs = obs.copy()
        history = np.concatenate(self.history).astype(np.float32)
        if self.name in ("ts", "waq"):
            return (obs[None], history[None])
        return ((obs if self.name == "simple" else history)[None],)

    def infer(self, *args):
        inputs = self.inputs(*args)
        with torch.inference_mode():
            result = self.module(*(torch.from_numpy(i) for i in inputs))
        self.action = result.numpy().reshape(12).copy()
        if self.name == "himloco":
            self.action = np.clip(self.action, -100, 100)
        if not np.isfinite(self.action).all():
            raise RuntimeError("Non-finite policy action")
        return self.default + self.action * self.scale


class Trial:
    period = 0.02

    def __init__(self, candidate: Candidate):
        self.policy = candidate
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        # Remove obstacle contact and visibility for a level-ground tracking test.
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == 0 and mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) != "floor":
                self.model.geom_contype[g] = self.model.geom_conaffinity[g] = 0
                self.model.geom_rgba[g, 3] = 0
        self.data = mujoco.MjData(self.model)
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[i])
               for i in candidate.mapping]
        if any(i < 0 for i in ids):
            raise ValueError("Missing joint")
        self.qadr = self.model.jnt_qposadr[ids]
        self.dadr = self.model.jnt_dofadr[ids]
        self.ranges = self.model.jnt_range[ids]
        self.substeps = round(self.period / self.model.opt.timestep)
        self.reset()

    def reset(self, yaw: float = 0, seed: int = 0):
        torch.manual_seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = (0, 0, .35, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        self.data.qpos[self.qadr] = self.policy.default
        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):
            self.advance(self.policy.default, 60, 5)
        self.policy.reset()
        self.elapsed = 0.0

    def advance(self, target, kp, kd):
        target = np.clip(target, self.ranges[:, 0], self.ranges[:, 1])
        for _ in range(self.substeps):
            tau = kp * (target - self.data.qpos[self.qadr]) - kd * self.data.qvel[self.dadr]
            self.data.ctrl[:] = 0
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.dadr] = np.clip(tau, -self.policy.torque_limit, self.policy.torque_limit)
            mujoco.mj_step(self.model, self.data)

    def step(self, command):
        q = self.data.qpos[self.qadr].astype(np.float32)
        dq = self.data.qvel[self.dadr].astype(np.float32)
        quat = self.data.qpos[3:7].astype(np.float32)
        gyro = self.data.qvel[3:6].astype(np.float32)
        target = self.policy.infer(q, dq, quat, gyro, command)
        self.advance(target, self.policy.kp, self.policy.kd)
        self.elapsed += self.period
        m = self.metrics()
        if not np.isfinite(self.data.qpos).all() or m["height"] < .15 or m["tilt"] > 45:
            raise RuntimeError(f"Fell at {self.elapsed:.2f}s: {m}")
        return m

    def metrics(self):
        w, x, y, z = self.data.qpos[3:7]
        return dict(x=float(self.data.qpos[0]), y=float(self.data.qpos[1]), height=float(self.data.qpos[2]),
                    yaw=float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))),
                    tilt=float(np.degrees(np.arccos(np.clip(1 - 2*(x*x+y*y), -1, 1)))),
                    speed=float(np.linalg.norm(self.data.qvel[:2])), yaw_rate=float(self.data.qvel[5]))


class FeedbackTrial(Trial):
    """Track an integrated Move target using simulated planar pose feedback."""

    def reset(self, yaw: float = 0, seed: int = 0):
        super().reset(yaw, seed)
        self.reference_xy = self.data.qpos[:2].copy()
        self.reference_yaw = self.metrics()["yaw"]
        self.yaw_integral = 0.0
        self.last_command = (0, 0, 0)

    def step(self, command):
        command = tuple(command)
        m = self.metrics()
        if command == (0, 0, 0) and self.last_command != (0, 0, 0):
            self.reference_xy = self.data.qpos[:2].copy()
        self.last_command = command
        self.reference_yaw += command[2] * self.period
        c, s = np.cos(self.reference_yaw), np.sin(self.reference_yaw)
        reference_rotation = np.array([[c, -s], [s, c]])
        world_velocity = reference_rotation @ np.array(command[:2])
        self.reference_xy += world_velocity * self.period
        c, s = np.cos(m['yaw']), np.sin(m['yaw'])
        world_to_body = np.array([[c, s], [-s, c]])
        corrected_xy = world_to_body @ (world_velocity + 1.2 * (self.reference_xy - self.data.qpos[:2]))
        yaw_error = np.arctan2(np.sin(self.reference_yaw - m['yaw']), np.cos(self.reference_yaw - m['yaw']))
        self.yaw_integral = float(np.clip(self.yaw_integral + yaw_error * self.period, -.3, .3))
        corrected_yaw = command[2] + 2.0 * yaw_error + .5*self.yaw_integral - .1*(m['yaw_rate']-command[2])
        corrected = (*np.clip(corrected_xy, -.8, .8), float(np.clip(corrected_yaw, -1, 1)))
        return super().step(corrected)


def run_case(trial, command=(.4, 0, 0), duration=10, yaw0=0, seed=0):
    trial.reset(yaw0, seed)
    initial = trial.metrics()
    history = []
    timings = []
    for _ in range(round(duration / trial.period)):
        t = time.perf_counter()
        history.append(trial.step(command))
        timings.append((time.perf_counter()-t)*1000)
    x = np.array([m['x'] - initial['x'] for m in history])
    y = np.array([m['y'] - initial['y'] for m in history])
    forward = x*np.cos(yaw0) + y*np.sin(yaw0)
    lateral = -x*np.sin(yaw0) + y*np.cos(yaw0)
    headings = np.unwrap([initial['yaw']] + [m['yaw'] for m in history])[1:] - initial['yaw']
    # Test the policy's own zero command, with no standing-hold workaround.
    stopped = [trial.step((0,0,0)) for _ in range(150)]
    stop_speed = float(np.mean([m['speed'] for m in stopped[-50:]]))
    stop_rotation = float(np.mean([abs(m['yaw_rate']) for m in stopped[-50:]]))
    result = dict(command=command, duration=duration, start_yaw=yaw0, seed=seed,
                  forward_m=float(forward[-1]), lateral_m=float(lateral[-1]),
                  max_lateral_m=float(np.max(np.abs(lateral))),
                  final_heading_deg=float(np.degrees(headings[-1])),
                  max_heading_deg=float(np.max(np.abs(np.degrees(headings)))),
                  max_tilt_deg=max(m['tilt'] for m in history+stopped),
                  stop_speed_m_s=stop_speed, stop_yaw_rate_rad_s=stop_rotation,
                  step_p95_ms=float(np.percentile(timings,95)))
    if command == (.4,0,0):
        result['straight_pass'] = result['max_lateral_m'] <= .10 and result['max_heading_deg'] <= 5
        result['speed_pass'] = 3.2 <= result['forward_m'] <= 4.8
        result['tracking_pass'] = result['straight_pass'] and result['speed_pass']
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policies', nargs='+', choices=CANDIDATES, default=CANDIDATES)
    parser.add_argument('--output', type=Path, default=TRIALS/'comparison.json')
    parser.add_argument('--feedback', action='store_true', help='enable simulated pose feedback; default tests raw policies')
    parser.add_argument('--gait', nargs=4, type=float, default=(.6,.34,.12,.3),
                        metavar=('PERIOD','HEIGHT','CLEARANCE','PITCH'))
    parser.add_argument('--extended', action='store_true', help='also test speeds, directions and rotated starts')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report=dict(platform=platform.platform(),mujoco=mujoco.__version__,torch=torch.__version__,
                gate='10s at vx=0.4: forward 3.2-4.8m, lateral <=0.10m, heading <=5 degrees',
                feedback=args.feedback, gait=args.gait,
                results={})
    for name in args.policies:
        try:
            trial_class=FeedbackTrial if args.feedback else Trial
            trial=trial_class(Candidate(name, args.gait))
            cases=[((.4,0,0),0)]
            if args.extended:
                cases += [((.4,0,0),.7), ((.4,0,0),-1.2), ((.2,0,0),0),
                          ((.6,0,0),0), ((-.3,0,0),0), ((0,.3,0),0), ((0,-.3,0),0),
                          ((0,0,.5),0), ((0,0,-.5),0), ((0,0,0),0)]
            result=[run_case(trial, command, yaw0=yaw) for command,yaw in cases]
        except Exception as exc:
            result=[{'error':str(exc),'tracking_pass':False}]
        report['results'][name]=result
        print(name, json.dumps(result),flush=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
