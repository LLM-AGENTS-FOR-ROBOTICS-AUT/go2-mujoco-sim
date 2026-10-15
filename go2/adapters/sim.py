"""Simulator adapter: owns the MuJoCo sim process and talks to it over local RPC."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from .. import procs
from ..perception import Mask
from .base import AdapterError, CameraFrame, RobotAdapter, RobotState

ROOT = Path(__file__).resolve().parents[2]
POSTURES = {"balance_stand": "standing", "walking": "walking", "animating": "busy", "lying": "lying",
            "sitting": "sitting", "locked_stand": "locked_stand", "damp": "damp", "holding": "holding",
            "fallen": "fallen"}
TRICKS = ("hello", "stretch", "content", "heart", "scrape", "dance1", "dance2")


class SimAdapter(RobotAdapter):
    kind = "sim"
    label = "MuJoCo simulation"
    requires_arming = False

    def __init__(self, python=None, scene="room", headless=False, detector="truth", world=None):
        if detector not in ("truth", "yolo"):
            raise ValueError("detector must be 'truth' or 'yolo'")
        self.python = Path(python) if python else Path(sys.executable)
        self.scene, self.headless, self.detector, self.world = scene, headless, detector, world
        self._yolo = None
        self.process = None
        self.info = None
        self.directory = ROOT / "runtime" / ("sim-" + uuid.uuid4().hex[:12])
        self.error = None

    # ---- process lifecycle -------------------------------------------------
    def connect(self, timeout=45):
        if self.connected:
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        status = self.directory / "status.json"
        status.unlink(missing_ok=True)
        interpreter = self.python
        if sys.platform == "darwin" and not self.headless:
            interpreter = self.python.parent / "mjpython"
            if not interpreter.is_file():
                raise AdapterError("mjpython was not found next to the selected Python; install requirements-simulation.txt")
        command = [str(interpreter), "-B", str(ROOT / "simulation/sim_server.py"), "--scene", self.scene,
                   "--status-file", str(status)] + (["--headless"] if self.headless else [])
        if self.world is not None:
            world_file = self.directory / "world.json"
            world_file.write_text(json.dumps(self.world))
            command += ["--world", str(world_file)]
        self.log = open(self.directory / "sim.log", "wb")
        self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log,
                                        **procs.detached())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = (self.directory / "sim.log").read_text(errors="replace")[-600:]
                self.process = None
                raise AdapterError("The simulator exited during startup. " + tail.strip().splitlines()[-1] if tail.strip() else "The simulator exited during startup.")
            try:
                self.info = json.loads(status.read_text())
                self.error = None
                if self.detector == "yolo":
                    self.warm_up()
                return
            except (OSError, ValueError):
                time.sleep(.1)
        self.close()
        raise AdapterError("The simulator did not become ready in time.")

    def close(self):
        process, self.process, self.info = self.process, None, None
        log, self.log = getattr(self, "log", None), None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                procs.stop_tree(process)
        if log is not None:
            log.close()
        if process is not None and self.error is None:
            shutil.rmtree(self.directory, ignore_errors=True)   # status file and log of a clean run

    @property
    def connected(self):
        return bool(self.process is not None and self.process.poll() is None and self.info)

    def status(self):
        value = super().status()
        value.update(scene=self.scene, headless=self.headless, detector=self.detector,
                     exited=self.process is not None and self.process.poll() is not None)
        return value

    # ---- RPC ---------------------------------------------------------------
    # Requests that are safe to send again if an answer is late (the simulator process can stall briefly
    # when the computer is busy); one-shot commands such as tricks are never repeated.
    RETRY = {"state": 2, "move": 1, "stop_move": 2, "lidar": 1}

    def _rpc(self, op, timeout=.5, **args):
        if not self.connected:
            raise AdapterError("The simulator is not running. Open MuJoCo first.")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.info['port']}/rpc", data=json.dumps({"op": op, "args": args}).encode(),
            headers={"Authorization": "Bearer " + self.info["token"], "Content-Type": "application/json"})
        for attempt in range(self.RETRY.get(op, 0) + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.load(response)["result"]
            except urllib.error.HTTPError as error:
                try:
                    message = json.load(error).get("error", str(error))
                except ValueError:
                    message = str(error)
                raise AdapterError(f"Simulator rejected {op}: {message}") from None
            except (OSError, TimeoutError) as error:
                if attempt == self.RETRY.get(op, 0) or not self.connected:
                    raise AdapterError(f"Simulator did not answer {op}: {error}") from None

    @staticmethod
    def _state(raw, stamp=None):
        return RobotState(stamp=stamp or time.monotonic(), x=raw["x"], y=raw["y"], yaw=raw["yaw"],
                          vx=raw["vx"], vy=raw["vy"], wz=raw["yaw_rate"], roll=raw["roll"], pitch=raw["pitch"],
                          height=raw["height"], posture=POSTURES.get(raw["mode"], raw["mode"]),
                          action=raw.get("action"), fault=raw.get("fault"),
                          extra={"sim_time": raw.get("sim_time"), "gait": raw.get("gait"),
                                 "real_time_factor": raw.get("real_time_factor")})

    def state(self):
        return self._state(self._rpc("state"))

    def move(self, vx, vy, wz):
        result = self._rpc("move", timeout=.5, vx=vx, vy=vy, wz=wz)
        if not result.get("ok"):
            raise AdapterError(result.get("error", "move refused"))

    def stop_move(self):
        self._rpc("stop_move", timeout=.5)

    def supports(self, name):
        if name in ("stand", "stand_up", "lie_down", "sit", "recovery_stand", "damp", "euler", "speed_level", "gait") or name in TRICKS:
            return None
        return f"{name.replace('_', ' ')} only runs on the real Go2; the simulator can't do acrobatics"

    def sport(self, name, **args):
        state = self.state()
        if name == "stand":
            if state.posture in ("standing", "walking"):
                return "already standing"
            call = ("rise_sit", {}) if state.posture == "sitting" else ("balance_stand", {})
        elif name in ("stand_up", "lie_down", "sit", "recovery_stand", "damp"):
            call = (name, {})
        elif name in TRICKS:
            call = ("trick", {"trick": name})
        elif name in ("euler", "speed_level", "gait"):
            call = (name, {"euler": {"roll": args.get("roll", 0.), "pitch": args.get("pitch", 0.), "yaw": args.get("yaw", 0.)},
                           "speed_level": {"level": args.get("level")}, "gait": {"gait": args.get("gait")}}[name])
        else:
            raise AdapterError(f"{name.replace('_', ' ')} is not available in the simulator; it only runs on the real Go2")
        result = self._rpc("sport", timeout=1.0, name=call[0], **call[1])
        if not result.get("ok"):
            raise AdapterError(result.get("error", "refused"))
        return result.get("note")

    def lidar(self):
        result = self._rpc("lidar", timeout=1.0)
        points = np.frombuffer(base64.b64decode(result["points"]), dtype=np.float32).reshape(-1, 3)
        return points.astype(float), self._state(result["state"])

    def warm_up(self):
        """Load YOLO in the background so the first look isn't slow (~10 s on first use)."""
        import threading
        def load():
            try:
                from ..detector import YoloSegDetector
                if self._yolo is None:
                    self._yolo = YoloSegDetector()
                self._yolo(np.zeros((720, 1280, 3), dtype=np.uint8))
            except Exception:
                pass   # reported on the first real look instead
        threading.Thread(target=load, daemon=True).start()

    def camera(self):
        yolo = self.detector == "yolo"
        result = self._rpc("camera", timeout=8.0, full=yolo)
        h, w = result["height"], result["width"]
        rgb = np.frombuffer(base64.b64decode(result["rgb"]), dtype=np.uint8).reshape(h, w, 3)
        if yolo:
            # The same detector as the real robot, on the rendered image (no ground truth).
            if self._yolo is None:
                from ..detector import YoloSegDetector
                self._yolo = YoloSegDetector()
            try:
                masks = self._yolo(rgb)
            except Exception as error:
                raise AdapterError(f"YOLO detection failed: {error}") from None
        else:
            index = np.frombuffer(base64.b64decode(result["index"]), dtype=np.uint8).reshape(h, w)
            masks = [Mask(item["label"], index == item["index"], 1.0) for item in result["objects"]]
        return CameraFrame(rgb=rgb, masks=masks, intrinsics=result["intrinsics"],
                           camera_pos=np.array(result["camera_pos"]), camera_rot=np.array(result["camera_rot"]),
                           state=self._state(result["state"]))

    def reset(self, x=None, y=None, yaw=None):
        pose = {} if x is None else {"x": x, "y": y or 0., "yaw": yaw or 0.}
        self._rpc("reset", timeout=3.0, **pose)
