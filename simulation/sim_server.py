"""Go2 simulation process: MuJoCo physics + optional viewer + authenticated local RPC.

Run with `mjpython` on macOS for the graphical viewer, or `python --headless`.
The app's SimAdapter is the only intended client. RPC is loopback-only and
requires the bearer token written to --token-file. Motion has a dead-man
timer: if no Move refresh arrives for 0.5 s the robot is told to StopMove.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import secrets
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]     # this folder, then the repo (go2_locomotion, policy_assets)
import numpy as np  # noqa: E402

from robot import SimGo2  # noqa: E402
from sensors import Camera, Lidar, camera_pose_in_base  # noqa: E402
from world import CAMERA_HEIGHT, CAMERA_WIDTH  # noqa: E402

DEADMAN_S = .5
SPORT = {  # RPC sport name -> SimGo2 call
    "stand_up": lambda s, a: s.StandUp(), "lie_down": lambda s, a: s.StandDown(),
    "balance_stand": lambda s, a: s.BalanceStand(), "recovery_stand": lambda s, a: s.RecoveryStand(),
    "sit": lambda s, a: s.Sit(), "rise_sit": lambda s, a: s.RiseSit(), "damp": lambda s, a: s.Damp(),
    "euler": lambda s, a: s.Euler(a["roll"], a["pitch"], a["yaw"]),
    "speed_level": lambda s, a: s.SpeedLevel(a["level"]), "gait": lambda s, a: s.SwitchGait(a["gait"]),
    "trick": lambda s, a: s.Trick(a["trick"]),
}


class Server:
    def __init__(self, scene, headless, world_spec=None):
        self.sim = SimGo2(scene, world_spec)
        self.headless = headless
        self.lidar = Lidar(self.sim.model)
        self.cameras = {}   # resolution -> Camera; GL contexts are thread-bound: only the render thread touches them
        self.render_requests = queue.Queue()
        threading.Thread(target=self._render_loop, daemon=True, name="render").start()
        self.last_move = None
        self.stopped = threading.Event()
        self.real_time_factor = 1.0
        self.started = time.monotonic()

    # ---- RPC operations ----------------------------------------------------
    def op(self, name, args):
        sim = self.sim
        if name == "state":
            return {**sim.state(), "real_time_factor": self.real_time_factor}
        if name == "move":
            with sim.lock:
                result = sim.Move(float(args["vx"]), float(args["vy"]), float(args["wz"]))
                if result["ok"]:
                    self.last_move = time.monotonic()
            return result
        if name == "stop_move":
            with sim.lock:
                self.last_move = None
                return sim.StopMove()
        if name == "sport":
            if args.get("name") not in SPORT:
                return {"ok": False, "error": "unknown sport command"}
            with sim.lock:
                self.last_move = None
                return SPORT[args["name"]](sim, args)
        if name == "lidar":
            with sim.lock:
                points = self.lidar.scan(sim.data)
                state = sim.state()
            return {"points": base64.b64encode(points.astype(np.float32).tobytes()).decode(),
                    "count": len(points), "frame": "base", "state": state}
        if name == "camera":
            return self.capture(args.get("full", False))
        if name == "reset":
            with sim.lock:
                self.last_move = None
                if "x" in args:
                    sim.reset((float(args["x"]), float(args.get("y", 0.)), float(args.get("yaw", 0.))))
                else:
                    sim.reset()
            return {"ok": True}
        raise ValueError("unknown operation")

    def _render_loop(self):
        while True:
            reply, box, full = self.render_requests.get()
            try:
                camera = self.cameras.get(full)
                if camera is None:
                    camera = self.cameras[full] = (Camera(self.sim.model, CAMERA_WIDTH, CAMERA_HEIGHT) if full
                                                   else Camera(self.sim.model))
                with self.sim.lock:  # copy the scene only; render outside the physics lock
                    state = self.sim.state()
                    camera.renderer.update_scene(self.sim.data, camera=camera.camera, scene_option=camera.options)
                rgb = camera.renderer.render().copy()
                camera.renderer.enable_segmentation_rendering()
                try:
                    seg = camera.renderer.render().copy()
                finally:
                    camera.renderer.disable_segmentation_rendering()
                box["value"] = (rgb, seg, state, camera.intrinsics())
            except Exception as error:
                box["error"] = error
            reply.set()

    def capture(self, full=False):
        reply, box = threading.Event(), {}
        self.render_requests.put((reply, box, bool(full)))
        if not reply.wait(10):
            raise TimeoutError("camera render timed out")
        if "error" in box:
            raise box["error"]
        rgb, seg, state, intrinsics = box["value"]
        ids, types = seg[..., 0], seg[..., 1]
        bodies = np.full(ids.shape, -1, dtype=np.int32)
        geoms = (types == 5) & (ids >= 0)  # mjOBJ_GEOM
        bodies[geoms] = self.sim.model.geom_bodyid[ids[geoms]]
        index = np.zeros(ids.shape, dtype=np.uint8)
        objects = []
        for body, item in self.sim.objects.items():
            if item.label == "wall":
                continue  # walls are obstacles for the LiDAR, not detector classes
            mask = bodies == body
            if mask.sum() >= 12:
                objects.append({"index": len(objects) + 1, "name": item.name, "label": item.label})
                index[mask] = len(objects)
        pos, rot = camera_pose_in_base(self.sim.model)
        return {"rgb": base64.b64encode(rgb.tobytes()).decode(), "index": base64.b64encode(index.tobytes()).decode(),
                "width": rgb.shape[1], "height": rgb.shape[0], "objects": objects, "intrinsics": intrinsics,
                "camera_pos": pos.tolist(), "camera_rot": rot.tolist(), "state": state}

    # ---- physics -----------------------------------------------------------
    def tick(self):
        with self.sim.lock:
            if self.last_move is not None and time.monotonic() - self.last_move > DEADMAN_S:
                self.sim.StopMove()
                self.last_move = None
            self.sim.step()

    def run(self, ready):
        period = self.sim.period
        viewer = None
        if not self.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data,
                                                  show_left_ui=False, show_right_ui=False)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 135, -35
            try:
                with viewer.lock():
                    # The viewer redraws continuously; shadows and reflections make that the costliest part of
                    # the simulator (the robot's own camera renders separately and keeps them). Flags set on
                    # user_scn are applied to the viewer's scene.
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            except (AttributeError, TypeError):
                pass   # only an optimisation; never stop the simulator over it
        ready()
        next_tick = time.monotonic()
        window, window_start = 0, time.monotonic()
        try:
            frame = 0
            while not self.stopped.is_set() and (viewer is None or viewer.is_running()):
                if viewer is not None:
                    # Physics never waits for the viewer's drawing (which slows down when the computer is busy):
                    # the viewer draws from its own copy, refreshed at 25 Hz by sync(). sync() copies mjData, so
                    # it stays under the physics lock that ray casts and camera renders also take.
                    with self.sim.lock:
                        self.tick()
                        if frame % 2 == 0:
                            with viewer.lock():
                                viewer.cam.lookat[:] = self.sim.data.qpos[self.sim.base_q:self.sim.base_q + 3]
                            viewer.sync()
                    frame += 1
                else:
                    self.tick()
                window += 1
                now = time.monotonic()
                if now - window_start >= 2:
                    self.real_time_factor = round(window * period / (now - window_start), 3)
                    window, window_start = 0, now
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -.25:  # fell far behind (e.g. machine paused); don't sprint to catch up
                    next_tick = time.monotonic()
        finally:
            with self.sim.lock:
                self.sim.StopMove()
            if viewer is not None:
                viewer.close()


def serve(server, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            code, result = 200, None
            try:
                if self.path != "/rpc" or self.headers.get("Authorization") != "Bearer " + token:
                    code = 403
                    raise PermissionError("unauthorized")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("bounded body required")
                request = json.loads(self.rfile.read(size))
                result = {"result": server.op(request["op"], request.get("args") or {})}
            except PermissionError as error:
                result = {"error": str(error)}
            except Exception as error:  # report, never crash the physics process
                code, result = 400, {"error": str(error)}
            payload = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("room", "empty", "procedural"), default="room")
    parser.add_argument("--world", help="JSON world description for --scene procedural")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--status-file", required=True, help="JSON written when ready: port, pid, token")
    parser.add_argument("--seconds", type=float, default=0, help="exit after this long (tests)")
    args = parser.parse_args()
    status = Path(args.status_file)
    token = secrets.token_hex(32)
    server = Server(args.scene, args.headless, json.loads(Path(args.world).read_text()) if args.world else None)
    httpd = serve(server, token)

    def halt(*_):
        server.stopped.set()
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    if args.seconds:
        threading.Timer(args.seconds, halt).start()
    parent = os.getppid()

    def watch_parent():
        # If the app that started us dies without closing us, we are re-parented: stop instead of lingering.
        while not server.stopped.is_set():
            time.sleep(1)
            if os.getppid() != parent:
                halt()
    threading.Thread(target=watch_parent, daemon=True, name="parent-watch").start()

    def ready():
        temporary = status.with_suffix(".new")
        temporary.write_text(json.dumps({"port": httpd.server_port, "pid": os.getpid(), "token": token,
                                         "scene": args.scene, "headless": args.headless}))
        temporary.chmod(0o600)
        temporary.replace(status)
    try:
        server.run(ready)
    finally:
        httpd.shutdown()
        status.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
