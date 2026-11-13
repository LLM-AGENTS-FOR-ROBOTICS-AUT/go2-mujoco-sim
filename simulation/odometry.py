"""Repeatable sensor odometry error; does not change physics or LiDAR rays.

Only this simulator sensor model sees truth poses. Localisation receives its
output, not the truth. Update every physics step, independently of RPC polling.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class DriftConfig:
    translation_scale: float = .02
    yaw_bias_per_m: float = .01
    translation_noise: float = .003  # metres / sqrt(metre travelled)
    yaw_noise: float = .002          # radians / sqrt(metre travelled)
    seed: int = 0

    def __post_init__(self):
        for key in ("translation_scale", "yaw_bias_per_m", "translation_noise", "yaw_noise"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or abs(value) > .2:
                raise ValueError(f"invalid odometry drift {key}")
        if self.translation_noise < 0 or self.yaw_noise < 0:
            raise ValueError("odometry noise must be nonnegative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("odometry seed must be a nonnegative integer")


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


class DriftingOdometry:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.config.seed)
        self.previous = self.pose = None

    def update(self, truth_pose):
        current = np.asarray(truth_pose, dtype=float)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("odometry pose must contain three finite coordinates")
        if self.previous is None:
            self.pose = current.copy()
        else:
            dx, dy = current[:2]-self.previous[:2]
            c, s = math.cos(self.previous[2]), math.sin(self.previous[2])
            body = np.array([c*dx+s*dy, -s*dx+c*dy])
            distance = float(np.linalg.norm(body))
            noise = math.sqrt(distance)
            body = body*(1+self.config.translation_scale) + self.rng.normal(0., self.config.translation_noise*noise, 2)
            c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
            self.pose[:2] += [c*body[0]-s*body[1], s*body[0]+c*body[1]]
            self.pose[2] = wrap(self.pose[2]+wrap(current[2]-self.previous[2])+
                                self.config.yaw_bias_per_m*distance+self.rng.normal(0., self.config.yaw_noise*noise))
        self.previous = current.copy()
        return self.pose.copy()

    def report(self):
        return {"pose": self.pose.tolist(), "config": asdict(self.config)}
