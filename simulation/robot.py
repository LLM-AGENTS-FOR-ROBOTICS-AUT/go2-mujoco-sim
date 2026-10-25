"""Simulated Go2 sport mode: a SportClient-shaped API over real MuJoCo physics.

Walking uses the pinned Walk-These-Ways policy with odometry feedback.
Postures and tricks are PD-controlled joint keyframe animations; body pose
(Euler) uses analytic leg inverse kinematics with the feet held in place.
Acrobatics (flips, jumps, handstand, ...) are not simulated and are refused.

All public methods are thread-safe and non-blocking. They return a dict with
"ok" plus "error" or "note". step() advances 20 ms of simulated time.
"""
from __future__ import annotations

import threading
import time

import mujoco
import numpy as np

import world
from go2_locomotion import JOINT_NAMES, WalkingPolicy
import go2_locomotion

L1 = L2 = .213
HIPS = np.array([[.1934, -.0465, 0], [.1934, .0465, 0], [-.1934, -.0465, 0], [-.1934, .0465, 0]])
OFFSETS = np.array([-.0955, .0955, -.0955, .0955])  # FR, FL, RR, RL lateral thigh offsets

STAND = np.tile([0.0, 0.8, -1.5], 4)
STAND_HIGH = np.array([-.0057, .6088, -1.2176, .0057, .6088, -1.2176] * 2)
LIE = np.array([-.0473, 1.2219, -2.4438, .0473, 1.2219, -2.4438] * 2)
SIT = np.array([-.05, .1, -1.05, .05, .1, -1.05, -.15, 2.2, -2.6, .15, 2.2, -2.6])

MOVE_LIMITS = np.array([.8, .8, 1.0])  # per-call limits are also enforced upstream
FEEDFORWARD = {"forward": .86, "backward": .79, "lateral": .75, "yaw": .68}
GAITS = {  # name -> (period s, body height, foot clearance, pitch); trot phase pattern
    "classic": (.5, .30, .08, 0.), "trot_run": (.4, .30, .09, 0.), "static_walk": (.6, .30, .06, 0.),
    "economic": (.5, .30, .06, 0.), "free_walk": (.5, .30, .08, 0.)}
ACROBATICS = {"front_flip", "back_flip", "left_flip", "front_jump", "front_pounce", "handstand",
              "walk_upright", "cross_step", "free_bound", "free_jump"}


def smoothstep(x):
    x = min(max(x, 0.), 1.)
    return x * x * (3 - 2 * x)


def leg_fk(q, i):
    q1, q2, q3 = q
    x = -L1 * np.sin(q2) - L2 * np.sin(q2 + q3)
    z = -L1 * np.cos(q2) - L2 * np.cos(q2 + q3)
    d = OFFSETS[i]
    return HIPS[i] + np.array([x, d * np.cos(q1) - z * np.sin(q1), d * np.sin(q1) + z * np.cos(q1)])


def leg_ik(foot, i):
    x, y, z = foot - HIPS[i]
    d = OFFSETS[i]
    L = np.sqrt(max(y * y + z * z - d * d, 1e-6))
    q1 = np.arctan2(z, y) + np.arctan2(L, d)
    q1 = np.arctan2(np.sin(q1), np.cos(q1))
    D2 = x * x + L * L
    c = np.clip((D2 - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1)
    q3 = -np.arccos(c)
    gamma = np.arctan2(-x, L)
    q2 = gamma + np.arctan2(L2 * np.sin(-q3), L1 + L2 * np.cos(q3))
    return np.array([q1, q2, q3])


NOMINAL_FEET = np.array([leg_fk(STAND[3 * i:3 * i + 3], i) for i in range(4)])


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def body_pose(roll=0., pitch=0., yaw=0., dz=0.):
    """Joint targets that rotate/raise the body while the feet stay planted."""
    r = rotation(roll, pitch, yaw)
    q = [leg_ik(r.T @ (NOMINAL_FEET[i] - np.array([0, 0, dz])), i) for i in range(4)]
    return np.concatenate(q)


class Animation:
    """Sequence of (duration, target, kp, kd); a target may be an array or f(t)->array."""

    def __init__(self, name, frames, end_mode, start):
        self.name, self.frames, self.end_mode = name, frames, end_mode
        self.index, self.elapsed, self.start = 0, 0., start.copy()

    def sample(self, dt):
        duration, target, kp, kd = self.frames[self.index]
        self.elapsed += dt
        if callable(target):
            goal = target(self.elapsed)
            alpha = smoothstep(self.elapsed / min(duration, .4))
        else:
            goal, alpha = target, smoothstep(self.elapsed / duration)
        q = (1 - alpha) * self.start + alpha * goal
        if self.elapsed >= duration:
            self.start, self.elapsed = goal.copy() if not callable(target) else target(duration), 0.
            self.index += 1
        return q, kp, kd

    @property
    def done(self):
        return self.index >= len(self.frames)


class SimGo2:
    period = .02

    def __init__(self, scene="room", world_spec=None):
        if scene == "house" and world_spec is None:
            import house
            world_spec = house.house_demo()          # start in its hallway, not the single room's start
        if scene == "hazards" and world_spec is None:
            import hazards
            world_spec = hazards.spec()              # start on the platform
        self.model, self.objects = world.build(scene, world_spec)
        self.start_pose = tuple(world_spec["start"]) if world_spec else world.START_POSE
        self.data = mujoco.MjData(self.model)
        self.policy = WalkingPolicy()
        self.substeps = round(self.period / self.model.opt.timestep)
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.qadr, self.dadr = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.ranges = self.model.jnt_range[ids]
        free = np.flatnonzero(self.model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))[0]
        self.base_q, self.base_d = int(self.model.jnt_qposadr[free]), int(self.model.jnt_dofadr[free])
        self.lock = threading.RLock()
        self.reset()

    # ----- lifecycle -------------------------------------------------------
    def reset(self, pose=None):
        with self.lock:
            x, y, yaw = pose or self.start_pose
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            z = .35 + self._ground_z(x, y, 5.)    # standing height above whatever is under the start (a platform)
            self.data.qpos[self.base_q:self.base_q + 7] = (x, y, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
            self.data.qpos[self.qadr] = STAND
            mujoco.mj_forward(self.model, self.data)
            for _ in range(100):
                self._advance(STAND, 60, 5)
            self.policy.reset()
            self.command = (0., 0., 0.)
            self.mode = "balance_stand"
            self.anim = None
            self.euler = (0., 0., 0.)
            self.gait = "classic"
            self.speed_level = 0
            self.fault = None
            self.walking = False
            self.hold = STAND.copy()

    # ----- SportClient-shaped commands ------------------------------------
    def _ok(self, note=None):
        return {"ok": True, **({"note": note} if note else {})}

    def _refuse(self, error):
        return {"ok": False, "error": error}

    def _gate(self, *modes):
        if self.mode == "fallen":
            return self._refuse("robot has fallen; use recovery_stand first")
        if self.anim is not None and self.anim.name == "stop_walk":
            # The post-walk settle is interruptible: treat the robot as balance standing.
            self.anim, self.mode, self.hold = None, "balance_stand", self.data.qpos[self.qadr].copy()
        if self.anim is not None and self.anim.name not in ("stop_walk",):
            return self._refuse(f"busy performing {self.anim.name}")
        if modes and self.mode not in modes:
            return self._refuse(f"not allowed while {self.mode.replace('_', ' ')}")
        return None

    def Move(self, vx, vy, vyaw):
        with self.lock:
            values = np.asarray([vx, vy, vyaw], dtype=float)
            if not np.isfinite(values).all() or (np.abs(values) > MOVE_LIMITS + 1e-9).any():
                return self._refuse("velocity outside limits (0.8 m/s, 1.0 rad/s)")
            settling = self.anim is not None and self.anim.name == "stop_walk"
            refused = None if settling else self._gate("balance_stand", "walking")
            if refused:
                return refused
            if settling:  # resume walking straight out of the stop blend
                self.anim = None
                self.mode = "balance_stand"
            self.command = tuple(float(v) for v in values)
            self.mode = "walking" if values.any() else self.mode
            return self._ok()

    def StopMove(self):
        with self.lock:
            self.command = (0., 0., 0.)
            if self.anim is not None and self.anim.name not in ("stop_walk",) and self.mode not in ("fallen",):
                # Stop also cancels a trick/posture in progress and holds the current pose.
                current = self.data.qpos[self.qadr].copy()
                self.anim = None
                self.hold = current
                self.mode = "holding"
            return self._ok()

    def _animate(self, name, frames, end_mode, hold):
        self.command = (0., 0., 0.)
        self.walking = False
        self.anim = Animation(name, frames, end_mode, self.data.qpos[self.qadr].copy())
        self._end_hold = hold
        self.mode = "animating"
        return self._ok()

    def BalanceStand(self):
        with self.lock:
            refused = self._gate("balance_stand", "walking", "locked_stand", "lying", "sitting", "holding", "damp")
            if refused:
                return refused
            if self.mode in ("balance_stand", "walking"):
                return self._ok("already balance standing")
            target = body_pose(*self.euler)
            frames = [(1.2, LIE, 60, 5), (1.0, target, 60, 5)] if self.mode == "damp" else [(1.0, target, 60, 5)]
            return self._animate("balance_stand", frames, "balance_stand", target)

    def StandUp(self):
        with self.lock:
            refused = self._gate("balance_stand", "walking", "lying", "sitting", "holding", "damp", "locked_stand")
            if refused:
                return refused
            if self.mode == "locked_stand":
                return self._ok("already standing")
            frames = [(1.2, LIE, 60, 5), (1.2, STAND_HIGH, 70, 5)] if self.mode == "damp" else [(1.2, STAND_HIGH, 70, 5)]
            return self._animate("stand_up", frames, "locked_stand", STAND_HIGH)

    def StandDown(self):
        with self.lock:
            refused = self._gate("balance_stand", "walking", "locked_stand", "sitting", "holding")
            if refused:
                return refused
            frames = [(1.0, STAND, 60, 5), (1.5, LIE, 60, 5)] if self.mode == "sitting" else [(1.5, LIE, 60, 5)]
            return self._animate("lie_down", frames, "lying", LIE)

    def Sit(self):
        with self.lock:
            refused = self._gate("balance_stand", "walking", "locked_stand", "holding")
            if refused:
                return refused
            return self._animate("sit", [(.6, STAND, 60, 5), (1.5, SIT, 70, 5)], "sitting", SIT)

    def RiseSit(self):
        with self.lock:
            refused = self._gate("sitting")
            if refused:
                return refused
            return self._animate("rise", [(1.5, STAND, 70, 5)], "balance_stand", STAND)

    def RecoveryStand(self):
        with self.lock:
            if self.anim is not None:
                return self._refuse(f"busy performing {self.anim.name}")
            note = None
            if self.mode == "fallen" or self._tilt() > 60:
                # Self-righting is not simulated: reposition upright in place, lying down.
                m = self.metrics()
                self.data.qpos[self.base_q:self.base_q + 7] = (m["x"], m["y"], .2, np.cos(m["yaw"] / 2), 0, 0, np.sin(m["yaw"] / 2))
                self.data.qvel[:] = 0
                self.data.qpos[self.qadr] = LIE
                mujoco.mj_forward(self.model, self.data)
                self.fault = None
                note = "simulated recovery: the robot was repositioned upright in place (self-righting is not simulated)"
            low = note is not None or self.mode in ("lying", "damp", "sitting")
            self.euler = (0., 0., 0.)
            frames = [(.5, LIE, 60, 5), (1.2, STAND_HIGH, 70, 5)] if low else [(1.0, STAND_HIGH, 70, 5)]
            result = self._animate("recovery_stand", frames, "locked_stand", STAND_HIGH)
            return self._ok(note) if result["ok"] else result

    def Damp(self):
        with self.lock:
            self.command = (0., 0., 0.)
            self.anim = None
            self.walking = False
            self.mode = "damp"
            return self._ok()

    def Euler(self, roll, pitch, yaw):
        with self.lock:
            if not all(np.isfinite([roll, pitch, yaw])) or abs(roll) > .75 or abs(pitch) > .75 or abs(yaw) > .6:
                return self._refuse("body angle outside limits (roll/pitch 0.75 rad, yaw 0.6 rad)")
            refused = self._gate("balance_stand")
            if refused:
                return refused
            self.euler = (float(roll), float(pitch), float(yaw))
            target = body_pose(*self.euler)
            return self._animate("body_pose", [(.8, target, 60, 5)], "balance_stand", target)

    def SpeedLevel(self, level):
        with self.lock:
            if level not in (-1, 0, 1):
                return self._refuse("speed level must be -1, 0 or 1")
            self.speed_level = level
            return self._ok("recorded; simulated walking speed follows the commanded velocity only")

    def SwitchGait(self, name):
        with self.lock:
            if name in ACROBATICS:
                return self._refuse(f"{name.replace('_', ' ')} is not simulated")
            if name not in GAITS:
                return self._refuse("unknown gait")
            refused = self._gate("balance_stand", "walking")
            if refused:
                return refused
            self.gait = name
            return self._ok()

    def Trick(self, name):
        with self.lock:
            if name in ACROBATICS:
                return self._refuse(f"{name.replace('_', ' ')} is not simulated")
            builder = getattr(self, "_trick_" + name, None)
            if builder is None:
                return self._refuse("unknown trick")
            refused = self._gate("balance_stand", "walking")
            if refused:
                return refused
            self.euler = (0., 0., 0.)
            return self._animate(name, builder(), "balance_stand", STAND)

    # ----- trick keyframes (approximations of Unitree's built-in motions) ---
    def _trick_hello(self):
        def wave(t):
            q = SIT.copy()
            q[0:3] = (-.35 + .25 * np.sin(2 * np.pi * 1.5 * t), -.9, -1.2)  # FR paw raised and waving
            return q
        return [(.6, STAND, 60, 5), (1.5, SIT, 70, 5), (2.5, wave, 70, 5), (.8, SIT, 70, 5), (1.5, STAND, 70, 5)]

    def _trick_stretch(self):
        bow = np.concatenate([leg_ik(NOMINAL_FEET[i] + (.08 if i < 2 else 0, 0, .14 if i < 2 else -.03), i) for i in range(4)])
        return [(1.2, bow, 60, 5), (1.5, bow, 60, 5), (1.2, STAND, 60, 5)]

    def _trick_content(self):
        return [(3.0, lambda t: body_pose(.18 * np.sin(2 * np.pi * 1.2 * t), -.08, .12 * np.sin(2 * np.pi * .6 * t)), 60, 5),
                (.8, STAND, 60, 5)]

    def _trick_heart(self):
        def paws(t):
            q = SIT.copy()
            lift = .15 * np.sin(2 * np.pi * t)
            q[0:3] = (.25, -.4 + lift, -1.6)
            q[3:6] = (-.25, -.4 + lift, -1.6)
            return q
        return [(.6, STAND, 60, 5), (1.5, SIT, 70, 5), (2.5, paws, 80, 5), (.8, SIT, 70, 5), (1.5, STAND, 70, 5)]

    def _trick_scrape(self):
        def greet(t):
            q = SIT.copy()
            bob = .25 * np.sin(2 * np.pi * 1.5 * t)
            q[0:3] = (.15, -.5 + bob, -1.4)
            q[3:6] = (-.15, -.5 + bob, -1.4)
            return q
        return [(.6, STAND, 60, 5), (1.5, SIT, 70, 5), (2.5, greet, 80, 5), (.8, SIT, 70, 5), (1.5, STAND, 70, 5)]

    def _trick_dance1(self):
        return [(6.0, lambda t: body_pose(.2 * np.sin(2 * np.pi * t), .1 * np.sin(4 * np.pi * t), 0, .03 * np.sin(4 * np.pi * t)), 60, 5),
                (.8, STAND, 60, 5)]

    def _trick_dance2(self):
        return [(6.0, lambda t: body_pose(.12 * np.sin(np.pi * t), .15 * np.cos(np.pi * t), .25 * np.sin(2 * np.pi * .5 * t)), 60, 5),
                (.8, STAND, 60, 5)]

    def _trick_wiggle_hips(self):
        return [(3.0, lambda t: body_pose(0, 0, .3 * np.sin(2 * np.pi * 1.5 * t)), 60, 5), (.8, STAND, 60, 5)]

    # ----- physics ----------------------------------------------------------
    def _advance(self, target, kp, kd):
        target = np.clip(target, self.ranges[:, 0], self.ranges[:, 1])
        for _ in range(self.substeps):
            torque = kp * (target - self.data.qpos[self.qadr]) - kd * self.data.qvel[self.dadr]
            self.data.ctrl[:] = 0
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.dadr] = np.clip(torque, -23.5, 23.5)
            mujoco.mj_step(self.model, self.data)

    def _reset_reference(self):
        self.reference_xy = self.data.qpos[self.base_q:self.base_q + 2].copy()
        self.reference_yaw = self.metrics()["yaw"]
        self.yaw_integral = 0.
        self.ramped = np.zeros(3)

    def _ramp(self):
        """Acceleration-limited reference velocity (1.6 m/s², 4 rad/s²)."""
        step = np.array([1.6, 1.6, 4.0]) * self.period
        self.ramped += np.clip(np.asarray(self.command) - self.ramped, -step, step)
        return self.ramped

    def _corrected_command(self):
        m = self.metrics()
        vx, vy, wz = self._ramp()
        self.reference_yaw += wz * self.period
        c, s = np.cos(self.reference_yaw), np.sin(self.reference_yaw)
        self.reference_xy += np.array([[c, -s], [s, c]]) @ np.array([vx, vy]) * self.period
        error = self.reference_xy - self.data.qpos[self.base_q:self.base_q + 2]
        if np.linalg.norm(error) > .5:
            raise RuntimeError("could not keep up with the commanded velocity")
        c, s = np.cos(m["yaw"]), np.sin(m["yaw"])
        body_error = np.array([[c, s], [-s, c]]) @ error
        ff = np.array([vx / (FEEDFORWARD["forward"] if vx >= 0 else FEEDFORWARD["backward"]), vy / FEEDFORWARD["lateral"]])
        xy = np.clip(ff + 1.2 * body_error, -1.3, 1.3)
        yaw_error = np.arctan2(np.sin(self.reference_yaw - m["yaw"]), np.cos(self.reference_yaw - m["yaw"]))
        self.yaw_integral = float(np.clip(self.yaw_integral + yaw_error * self.period, -.3, .3))
        yaw = wz / FEEDFORWARD["yaw"] + 2 * yaw_error + .5 * self.yaw_integral - .1 * (m["yaw_rate"] - wz)
        return (*xy, float(np.clip(yaw, -2., 2.)))

    def step(self):
        with self.lock:
            if self.mode == "fallen":
                self._advance(self.data.qpos[self.qadr], 0, 1.)
                return
            try:
                speed = np.hypot(*self.data.qvel[self.base_d:self.base_d + 2])
                braking = self.walking and (np.abs(self.ramped).max() > 1e-3 or speed > .12)
                if self.mode == "walking" and (any(self.command) or braking):
                    # Zero commands brake through the gait (ramped reference) before standing.
                    if not self.walking:
                        self.policy.reset()
                        self._reset_reference()
                        self.walking = True
                    go2_locomotion.GAIT = GAITS[self.gait]
                    q = self.data.qpos[self.qadr].astype(np.float32)
                    target = self.policy.step(
                        q, self.data.qvel[self.dadr].astype(np.float32),
                        self.data.qpos[self.base_q + 3:self.base_q + 7].astype(np.float32),
                        self.data.qvel[self.base_d + 3:self.base_d + 6].astype(np.float32), self._corrected_command())
                    self._advance(target, 20, .5)
                elif self.mode == "walking":
                    # Command went to zero: blend smoothly back into a balanced stand.
                    self.walking = False
                    self.anim = Animation("stop_walk", [(.5, body_pose(*self.euler), 40, 3)], "balance_stand",
                                          self.data.qpos[self.qadr].copy())
                    self._end_hold = body_pose(*self.euler)
                    self.mode = "animating"
                    q, kp, kd = self.anim.sample(self.period)
                    self._advance(q, kp, kd)
                elif self.mode == "animating" and self.anim is not None:
                    q, kp, kd = self.anim.sample(self.period)
                    self._advance(q, kp, kd)
                    if self.anim.done:
                        self.mode, self.hold, self.anim = self.anim.end_mode, self._end_hold, None
                elif self.mode == "damp":
                    self._advance(self.data.qpos[self.qadr], 0, 1.5)
                else:
                    target = {"lying": LIE, "sitting": SIT, "locked_stand": STAND_HIGH}.get(self.mode, self.hold)
                    self._advance(target, 70 if self.mode == "sitting" else 60, 5)
            except RuntimeError as error:
                self._fall(str(error))
                return
            m = self.metrics()
            limit = 75 if self.mode in ("sitting", "animating") else 55
            upright_mode = self.mode in ("walking", "balance_stand", "locked_stand")
            if not np.isfinite(self.data.qpos).all():
                self.reset()
                self.fault = "simulation became unstable and was reset"
            elif self.mode not in ("damp", "lying") and (m["tilt_deg"] > limit or (upright_mode and m["height"] < .08)):
                self._fall(f"fell over (tilt {m['tilt_deg']:.0f}°, height {m['height']:.2f} m)")

    def _fall(self, reason):
        self.command = (0., 0., 0.)
        self.anim = None
        self.walking = False
        self.fault = reason
        if self._tilt() > 55:
            self.mode = "fallen"
        else:  # e.g. a tracking fault: settle back into a balanced stand
            self.anim = Animation("stop_walk", [(.5, STAND, 40, 3)], "balance_stand", self.data.qpos[self.qadr].copy())
            self._end_hold = STAND.copy()
            self.euler = (0., 0., 0.)
            self.mode = "animating"

    GROUND_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)    # what the LiDAR sees; glass (group 4) isn't ground

    def _ground_z(self, x, y, top):
        """World z of the first surface below (x, y, top) that isn't part of the robot; 0 if there is none."""
        root = self.model.body_rootid
        robot = root[self.model.jnt_bodyid[np.flatnonzero(self.model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))[0]]]
        origin, down, geom = np.array([x, y, top], float), np.array([0., 0., -1.]), np.zeros(1, np.int32)
        for _ in range(20):
            dist = mujoco.mj_ray(self.model, self.data, origin, down, self.GROUND_GROUPS, 1, -1, geom)
            if geom[0] < 0:
                return 0.
            origin = origin + down * dist
            if root[self.model.geom_bodyid[geom[0]]] != robot:
                return float(origin[2])
            origin[2] -= 1e-4                                  # past one of the robot's own geoms, keep looking
        return 0.

    def _tilt(self):
        return self.metrics()["tilt_deg"]

    def metrics(self):
        qa, da = self.base_q, self.base_d
        w, x, y, z = self.data.qpos[qa + 3:qa + 7]
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = self.data.qvel[da:da + 2]
        bx, by, bz = (float(v) for v in self.data.qpos[qa:qa + 3])
        height = bz - self._ground_z(bx, by, bz)            # above the ground under the robot, as on the real Go2
        return dict(x=bx, y=by, height=height,
                    yaw=yaw, roll=float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
                    pitch=float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))),
                    tilt_deg=float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))),
                    vx=float(c * vx + s * vy), vy=float(-s * vx + c * vy), yaw_rate=float(self.data.qvel[da + 5]))

    def state(self):
        with self.lock:
            m = self.metrics()
            return {**m, "sim_time": float(self.data.time), "mode": self.mode,
                    "action": self.anim.name if self.anim else None, "command": list(self.command),
                    "gait": self.gait, "euler": list(self.euler), "speed_level": self.speed_level,
                    "fault": self.fault, "stamp": time.monotonic()}
