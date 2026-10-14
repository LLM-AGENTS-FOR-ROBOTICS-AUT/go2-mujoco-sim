"""Simulated Go2 sensors matching the real hardware's published specs.

LiDAR: Unitree 4D LiDAR L1 — 360° x 90° field of view (a hemisphere above the
sensor's own equator), 0.05 m blind zone, 30 m range at 90% reflectivity,
±2 cm accuracy, 21600 effective points/s with an 11 Hz circumferential scan
(~1960 points per scan) and a non-repetitive pattern. It is mounted upside
down and pitched forward on the Go2's chin (URDF radar_joint), so it sees the
floor all around and up to ~15° above horizontal ahead.

Camera: front wide-angle camera, 1280x720, 120° horizontal FOV.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from world import CAMERA_HEIGHT, CAMERA_WIDTH

LIDAR_POINTS_PER_SCAN = 1964
LIDAR_MIN_RANGE, LIDAR_MAX_RANGE, LIDAR_NOISE_STD = .05, 30.0, .01
GOLDEN = (math.sqrt(5) - 1) / 2


class Lidar:
    def __init__(self, model, seed=0):
        self.model = model
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        if self.site < 0:
            raise ValueError("World has no lidar site")
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.rng = np.random.default_rng(seed)
        # World geoms (groups 0/1) return points; the robot's own visual meshes (group 2) block
        # beams like the real body does, and those self-hits are dropped as the real driver does.
        self.geomgroup = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        self.robot_root = model.body_rootid[self.base]

    def _directions(self):
        """Non-repetitive hemisphere pattern in the sensor frame (z = sensor pole)."""
        n = LIDAR_POINTS_PER_SCAN
        k = np.arange(n)
        # Fibonacci hemisphere, re-offset every scan so coverage fills in over time.
        azimuth = 2 * np.pi * ((k * GOLDEN + self.rng.random()) % 1.0)
        # Uniform over solid angle between elevation 0 (equator) and 90° (pole).
        sin_elevation = (k + self.rng.random()) / n
        cos_elevation = np.sqrt(1 - sin_elevation ** 2)
        return np.stack([cos_elevation * np.cos(azimuth), cos_elevation * np.sin(azimuth), sin_elevation], 1)

    def scan(self, data):
        """Return hit points in the robot base frame (x forward, y left, z up)."""
        directions = self._directions()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        origin = data.site_xpos[self.site].copy()
        world_dirs = directions @ rotation.T
        n = len(world_dirs)
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0)
        # The site sits inside the chin bulge that models the L1 itself, and the torso is above the
        # dome, so the base body is excluded; the legs still block beams like the real ones do.
        # No cutoff here: mj_multiRay culls geoms by their position, and the infinite floor sits at the world
        # origin, so a 30 m cutoff made the floor vanish more than 30 m from the origin. Range is applied below.
        mujoco.mj_multiRay(self.model, data, origin, world_dirs.reshape(-1), self.geomgroup, 1, self.base,
                           geomid, dist, None, n, mujoco.mjMAXVAL)
        own = (geomid >= 0) & (self.model.body_rootid[self.model.geom_bodyid[np.maximum(geomid, 0)]] == self.robot_root)
        hit = (geomid >= 0) & ~own & (dist >= LIDAR_MIN_RANGE) & (dist <= LIDAR_MAX_RANGE)
        ranges = dist[hit] + self.rng.normal(0, LIDAR_NOISE_STD, hit.sum())
        points_world = origin + world_dirs[hit] * ranges[:, None]
        base_pos = data.xpos[self.base]
        base_rot = data.xmat[self.base].reshape(3, 3)
        return ((points_world - base_pos) @ base_rot).astype(np.float32)


class Camera:
    """Offscreen renderer for the front camera: RGB plus per-pixel world body ids."""

    def __init__(self, model, width=CAMERA_WIDTH // 2, height=CAMERA_HEIGHT // 2):
        self.model = model
        self.width, self.height = width, height
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
        if self.camera < 0:
            raise ValueError("World has no front_camera")
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.options = mujoco.MjvOption()
        for group in (2, 3):  # the camera sits inside the head shell; never render the robot
            self.options.geomgroup[group] = 0
        self.options.geomgroup[5] = 1   # room ceiling: camera-only
        for group in (4, 5):
            self.options.sitegroup[group] = 0

    def intrinsics(self):
        fovy = math.radians(self.model.cam_fovy[self.camera])
        fy = self.height / 2 / math.tan(fovy / 2)
        return {"fx": fy, "fy": fy, "cx": self.width / 2, "cy": self.height / 2,
                "width": self.width, "height": self.height}

    def capture(self, data):
        """Return (rgb uint8 HxWx3, body id map int32 HxW with -1 for background)."""
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.options)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(data, camera=self.camera, scene_option=self.options)
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        ids, types = seg[..., 0], seg[..., 1]
        bodies = np.full(ids.shape, -1, dtype=np.int32)
        geoms = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        bodies[geoms] = self.model.geom_bodyid[ids[geoms]]
        return rgb, bodies

    def close(self):
        self.renderer.close()


def camera_pose_in_base(model):
    """Camera position and rotation (camera->base, OpenCV axes: x right, y down, z forward)."""
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    pos = model.cam_pos[camera].copy()
    # OpenCV optical frame expressed in base axes: z fwd = +x, x right = -y, y down = -z
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    return pos, rotation
