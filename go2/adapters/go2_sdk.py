"""Real Go2 EDU adapter over the official unitree_sdk2py (DDS over Ethernet).

Requires `unitree_sdk2py` (with CycloneDDS) and, for object detection, the
`ultralytics` package plus YOLO segmentation weights. Imports are lazy so the
rest of the app runs without them.

Assumptions to verify on the robot (marked VERIFY below):
  * SportModeState.velocity is in the odom frame (converted to body frame here).
  * SportModeState.mode numbering (see MODES).
  * Euler pitch sign: positive pitch lowers the nose (tool layer sends look-up as negative).
  * LiDAR topic/frame: rt/utlidar/cloud_deskewed in "odom", or rt/utlidar/cloud in the sensor frame.
  * Front camera intrinsics: pinhole approximation of the 120° wide-angle lens.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from ..perception import base_to_level, rotation_rp
from .base import AdapterError, CameraFrame, RobotAdapter, RobotState

ROOT = Path(__file__).resolve().parents[2]
# VERIFY: Go2 SportModeState.mode values.
MODES = {0: "standing", 1: "standing", 2: "holding", 3: "walking", 5: "lying", 6: "locked_stand",
         7: "damp", 8: "busy", 10: "sitting", 11: "busy", 12: "busy", 13: "busy"}
LIDAR_MOUNT_POS = np.array([.28945, 0., -.046825])
LIDAR_MOUNT_PITCH = 2.8782
CAMERA_POS = np.array([.32715, -.00003, .04297])
CAMERA_ROT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # OpenCV axes in base axes
TRICKS = {"hello": "Hello", "stretch": "Stretch", "content": "Content", "heart": "Heart", "scrape": "Scrape",
          "dance1": "Dance1", "dance2": "Dance2"}
ONE_SHOT = {"front_flip": "FrontFlip", "back_flip": "BackFlip", "left_flip": "LeftFlip",
            "front_jump": "FrontJump", "front_pounce": "FrontPounce"}
MODE_TOGGLES = {"handstand": "HandStand", "walk_upright": "WalkUpright", "cross_step": "CrossStep",
                "free_bound": "FreeBound", "free_jump": "FreeJump"}
GAITS = {"classic": ("ClassicWalk", True), "trot_run": ("TrotRun", None), "static_walk": ("StaticWalk", None),
         "free_walk": ("FreeWalk", None)}


def _pointcloud_xyz(msg):
    fields = {f.name: f for f in msg.fields}
    if not {"x", "y", "z"} <= set(fields):
        return np.zeros((0, 3))
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    count = len(raw) // msg.point_step
    raw = raw[:count * msg.point_step].reshape(count, msg.point_step)
    order = ">" if msg.is_bigendian else "<"
    xyz = [raw[:, fields[k].offset:fields[k].offset + 4].copy().view(order + "f4").reshape(-1) for k in "xyz"]
    points = np.stack(xyz, 1).astype(float)
    return points[np.isfinite(points).all(1)]


class Go2Adapter(RobotAdapter):
    kind = "go2"
    label = "Real Go2 EDU"
    requires_arming = True

    def __init__(self, interface, lidar_topic="rt/utlidar/cloud_deskewed", yolo_weights=None, camera_hfov_deg=120.):
        self.interface = interface
        self.lidar_topic = lidar_topic
        self.yolo_weights = yolo_weights   # None -> GO2_YOLO_WEIGHTS or models/yolo11n-seg.pt
        self.camera_hfov = camera_hfov_deg
        self.client = self.video = self.detector = None
        self._sport_state = self._battery = self._cloud = None
        self._lock = threading.Lock()
        self._last_move = None
        self._watchdog = None
        self._closed = threading.Event()
        self._subscribers = []
        self._active_modes = set()

    # ---- lifecycle -----------------------------------------------------------
    def connect(self):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.go2.video.video_client import VideoClient
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
        except ImportError as error:
            raise AdapterError("unitree_sdk2py is not installed in this Python environment "
                               f"({error}). Install it with CycloneDDS to drive the real Go2.") from None
        if not self.interface:
            raise AdapterError("Choose the network interface connected to the Go2 (e.g. en7).")
        ChannelFactoryInitialize(0, self.interface)
        for topic, kind, handler in (("rt/sportmodestate", SportModeState_, self._on_sport),
                                     ("rt/lowstate", LowState_, self._on_low),
                                     (self.lidar_topic, PointCloud2_, self._on_cloud)):
            subscriber = ChannelSubscriber(topic, kind)
            subscriber.Init(handler, 10)
            self._subscribers.append(subscriber)
        self.client = SportClient()
        self.client.SetTimeout(5.0)
        self.client.Init()
        self.video = VideoClient()
        self.video.SetTimeout(3.0)
        self.video.Init()
        deadline = time.monotonic() + 5
        while self._sport_state is None and time.monotonic() < deadline:
            time.sleep(.05)
        if self._sport_state is None:
            raise AdapterError(f"No sport state from the Go2 on {self.interface} within 5 s. "
                               "Check the cable, the 192.168.123.x address and that sport mode is running.")
        self._closed.clear()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def close(self):
        self._closed.set()
        if self.client is not None:
            try:
                self.client.StopMove()
            except Exception:
                pass
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass
        self._subscribers = []
        self.client = None

    @property
    def connected(self):
        return self.client is not None and self._sport_state is not None and \
            time.monotonic() - self._sport_state[0] < 1.0

    def status(self):
        value = super().status()
        value.update(interface=self.interface, detector="yolo")
        return value

    # ---- DDS callbacks -------------------------------------------------------
    def _on_sport(self, msg):
        with self._lock:
            self._sport_state = (time.monotonic(), msg)

    def _on_low(self, msg):
        try:
            self._battery = float(msg.bms_state.soc)
        except AttributeError:
            pass

    def _on_cloud(self, msg):
        with self._lock:
            self._cloud = (time.monotonic(), msg)

    def _watch(self):
        """Stop the robot if Move refreshes stop arriving (the app or executor stalled)."""
        while not self._closed.wait(.1):
            last = self._last_move
            if last is not None and time.monotonic() - last > .4:
                self._last_move = None
                try:
                    self.client.StopMove()
                except Exception:
                    pass

    # ---- state ---------------------------------------------------------------
    def state(self):
        with self._lock:
            sample = self._sport_state
        if sample is None:
            raise AdapterError("No state received from the Go2.")
        stamp, msg = sample
        roll, pitch, yaw = (float(v) for v in msg.imu_state.rpy)
        vxw, vyw = float(msg.velocity[0]), float(msg.velocity[1])   # VERIFY: odom-frame velocity
        c, s = math.cos(yaw), math.sin(yaw)
        posture = MODES.get(int(msg.mode), "unknown")
        if posture == "walking" and math.hypot(vxw, vyw) < .03 and abs(float(msg.yaw_speed)) < .05:
            posture = "standing"
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            posture = "fallen"
        return RobotState(stamp=stamp, x=float(msg.position[0]), y=float(msg.position[1]), yaw=yaw,
                          vx=c * vxw + s * vyw, vy=-s * vxw + c * vyw, wz=float(msg.yaw_speed), roll=roll,
                          pitch=pitch, height=float(msg.body_height) or .3, posture=posture,
                          fault=None if not msg.error_code else f"sport error code {msg.error_code}",
                          battery_pct=self._battery, extra={"mode": int(msg.mode), "gait_type": int(msg.gait_type)})

    # ---- commands --------------------------------------------------------------
    def _call(self, method, *args):
        if self.client is None:
            raise AdapterError("The Go2 is not connected.")
        code = getattr(self.client, method)(*args)
        if code not in (0, None):
            raise AdapterError(f"The Go2 rejected {method} (code {code}).")

    def move(self, vx, vy, wz):
        self._call("Move", float(vx), float(vy), float(wz))
        self._last_move = time.monotonic()

    def stop_move(self):
        self._last_move = None
        self._call("StopMove")

    def sport(self, name, **args):
        posture = self.state().posture
        if name == "stand":
            if posture in ("standing", "walking"):
                return "already standing"
            if posture == "sitting":
                self._call("RiseSit")
            elif posture in ("lying", "damp", "fallen"):
                self._call("RecoveryStand" if posture != "lying" else "StandUp")
                time.sleep(2.0)
                self._call("BalanceStand")
            else:
                self._call("BalanceStand")
            return None
        simple = {"stand_up": "StandUp", "lie_down": "StandDown", "sit": "Sit", "recovery_stand": "RecoveryStand",
                  "damp": "Damp", **TRICKS, **ONE_SHOT}
        if name in simple:
            self._call(simple[name])
        elif name in MODE_TOGGLES:
            self._call(MODE_TOGGLES[name], True)
            self._active_modes.add(name)
        elif name == "euler":
            self._call("Euler", args.get("roll", 0.), args.get("pitch", 0.), args.get("yaw", 0.))  # VERIFY sign
        elif name == "speed_level":
            self._call("SpeedLevel", int(args["level"]))
        elif name == "gait":
            gait = args.get("gait")
            if gait not in GAITS:
                raise AdapterError(f"The {gait} gait is not exposed by this SDK version.")
            for mode in sorted(self._active_modes):   # leave handstand, walk-upright etc. first
                self._call(MODE_TOGGLES[mode], False)
            self._active_modes.clear()
            method, flag = GAITS[gait]
            self._call(method, *(() if flag is None else (flag,)))
        else:
            raise AdapterError(f"Unknown Go2 command {name}.")
        return None

    # ---- sensors -----------------------------------------------------------------
    def lidar(self):
        with self._lock:
            cloud = self._cloud
        if cloud is None or time.monotonic() - cloud[0] > 1.0:
            raise AdapterError("No recent LiDAR data from the Go2.")
        state = self.state()
        points = _pointcloud_xyz(cloud[1])
        frame = str(cloud[1].header.frame_id)
        if "odom" in frame or "world" in frame:
            c, s = math.cos(state.yaw), math.sin(state.yaw)
            dx, dy = points[:, 0] - state.x, points[:, 1] - state.y
            level = np.stack([c * dx + s * dy, -s * dx + c * dy, points[:, 2] - state.height], 1)
            base = level @ rotation_rp(state.roll, state.pitch)
        else:  # sensor frame -> base via the URDF radar_joint mount
            cp, sp = math.cos(LIDAR_MOUNT_PITCH), math.sin(LIDAR_MOUNT_PITCH)
            rot = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            base = points @ rot.T + LIDAR_MOUNT_POS
        own_body = (np.abs(base[:, 0]) < .45) & (np.abs(base[:, 1]) < .25) & (base[:, 2] > -.45) & (base[:, 2] < .2)
        keep = ~own_body   # drop returns from the robot's own legs and body
        return base[keep], state

    def _detector(self):
        if self.detector is None:
            from ..detector import YoloSegDetector
            self.detector = YoloSegDetector(self.yolo_weights)
        return self.detector

    def camera(self):
        if self.video is None:
            raise AdapterError("The Go2 camera is not connected.")
        code, data = self.video.GetImageSample()
        if code != 0 or not data:
            raise AdapterError(f"The Go2 camera returned no image (code {code}).")
        import cv2
        bgr = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = bgr[..., ::-1].copy()
        h, w = rgb.shape[:2]
        fx = w / 2 / math.tan(math.radians(self.camera_hfov) / 2)   # VERIFY: calibrate the wide-angle lens
        try:
            masks = self._detector()(rgb)
        except Exception as error:
            raise AdapterError(f"Object detection failed: {error}") from None
        return CameraFrame(rgb=rgb, masks=masks,
                           intrinsics={"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2, "width": w, "height": h},
                           camera_pos=CAMERA_POS, camera_rot=CAMERA_ROT, state=self.state())


def interfaces():
    """Network interfaces that could reach the Go2 (192.168.123.0/24 is Unitree's default)."""
    import socket
    names = []
    try:
        for _, name in socket.if_nameindex():
            if name.startswith(("en", "eth", "enp", "bridge")):
                names.append(name)
    except OSError:
        pass
    return names


__all__ = ["Go2Adapter", "interfaces", "base_to_level"]
