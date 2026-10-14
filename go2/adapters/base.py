"""One robot interface for the executor; implemented by the simulator and the real Go2.

Adapters speak SportClient semantics. Sport names are the tool-level names:
postures (stand, stand_up, lie_down, sit, recovery_stand, damp), tricks
(hello, stretch, content, heart, scrape, dance1, dance2), acrobatics
(front_flip, ..., free_jump; modes end on a gait change), and body settings
(euler, speed_level, gait).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# Normalised postures reported by every adapter: "standing" (balanced, can walk), "walking", "busy" (a transition,
# trick or acrobatic in progress), "fallen", "unknown", or one of these low ones:
LOW_POSTURES = ("lying", "sitting", "locked_stand", "damp", "holding")


class AdapterError(RuntimeError):
    """The robot refused or could not perform a command; message is user-presentable."""


@dataclass
class RobotState:
    stamp: float                 # time.monotonic() when this state was observed
    x: float
    y: float
    yaw: float
    vx: float = 0.
    vy: float = 0.
    wz: float = 0.
    roll: float = 0.
    pitch: float = 0.
    height: float = .3
    posture: str = "unknown"
    action: str | None = None
    fault: str | None = None
    battery_pct: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def age(self):
        return time.monotonic() - self.stamp

    @property
    def speed(self):
        return float(np.hypot(self.vx, self.vy))


@dataclass
class CameraFrame:
    rgb: np.ndarray
    masks: list
    intrinsics: dict
    camera_pos: np.ndarray       # camera position in the base frame
    camera_rot: np.ndarray       # OpenCV camera axes expressed in base axes (3x3)
    state: RobotState


class RobotAdapter(ABC):
    kind = "abstract"
    label = "robot"
    requires_arming = False

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def close(self): ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    @abstractmethod
    def state(self) -> RobotState: ...

    @abstractmethod
    def move(self, vx, vy, wz):
        """Body-frame velocity command; must be refreshed at least every 0.3 s while moving."""

    @abstractmethod
    def stop_move(self): ...

    @abstractmethod
    def sport(self, name, **args) -> str | None:
        """Start a posture/trick/acrobatic/body command. Returns an optional note; raises AdapterError."""

    @abstractmethod
    def lidar(self) -> tuple[np.ndarray, RobotState]:
        """Latest LiDAR points in the base frame and the state they were captured at."""

    @abstractmethod
    def camera(self) -> CameraFrame: ...

    def supports(self, name) -> str | None:
        """None if the sport command is available, else a user-presentable reason it is not."""
        return None

    def status(self) -> dict:
        return {"kind": self.kind, "label": self.label, "connected": self.connected}
