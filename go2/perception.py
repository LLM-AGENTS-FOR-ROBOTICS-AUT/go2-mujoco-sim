"""Shared perception: fuse segmentation masks with the LiDAR cloud, and LiDAR clearance.

Used identically for the simulator (ground-truth segmentation) and the real Go2
(YOLO segmentation). Frames:
  base   x forward, y left, z up, rigidly attached to the robot body
  level  base rotated by roll/pitch back to gravity-aligned (yaw kept)
  odom   adapter odometry frame (x, y, yaw of the body)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

# Go2 body footprint (m): length 0.70, width 0.31, standing height ~0.40
HALF_LENGTH, HALF_WIDTH = .36, .17
OBSTACLE_MIN_Z, OBSTACLE_MAX_Z = .06, .55   # height above the floor that can block the body
# A return this far below the robot's own floor is a drop (a ledge, stairs going down): the same 6 cm the
# obstacle check uses going up. Indoors a downward ramp also looks like a drop, which is the safe side.
STEP_DOWN_Z = .06
# Beyond this the floor-contact estimate is unreliable (one pixel is metres of distance): say "unknown".
GROUND_PLANE_MAX_M = 6.
NO_COLOR = {"person", "wall"}   # a person's "colour" is just their clothes; don't name people by it
SECTORS = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right")


@dataclass
class Mask:
    label: str
    mask: np.ndarray        # HxW bool
    score: float = 1.0


@dataclass
class Detection:
    id: str
    label: str
    color: str | None
    distance_m: float | None    # horizontal distance from the body centre to the nearest surface
    bearing_deg: float          # positive = to the robot's left
    x_m: float | None           # level frame position of the object's near surface
    y_m: float | None
    distance_source: str        # "lidar", "ground_plane" or "unknown"
    confidence: float
    # Internal only (not shown to the model, whose look results stay as trained): a person's shirt colour, used
    # to tell people apart while following, and where the object is in the image as [left, top, right, bottom].
    clothing: str | None = None
    box: list | None = None

    def public(self):
        value = asdict(self)
        value.pop("clothing")
        value.pop("box")
        for key in ("distance_m", "x_m", "y_m"):
            if value[key] is not None:
                value[key] = round(value[key], 2)
        value["bearing_deg"] = round(value["bearing_deg"])
        value["confidence"] = round(value["confidence"], 2)
        return value


def rotation_rp(roll, pitch):
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


def base_to_level(points, roll, pitch):
    return points @ rotation_rp(roll, pitch).T


def level_to_odom(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = points.copy()
    out[:, 0] = x + c * points[:, 0] - s * points[:, 1]
    out[:, 1] = y + s * points[:, 0] + c * points[:, 1]
    return out


def odom_to_level(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = points[:, 0] - x, points[:, 1] - y
    out = points.copy()
    out[:, 0] = c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    return out


class CloudBuffer:
    """Accumulates recent scans in the odom frame (the L1 pattern fills in over time)."""

    def __init__(self, keep_s=.6):
        self.keep_s = keep_s
        self.scans = []   # (stamp, odom points Nx3 with z measured from the floor)

    def add(self, stamp, points_level, pose, height):
        pts = level_to_odom(points_level, pose)
        pts[:, 2] += height
        self.scans.append((stamp, pts))
        self.scans = [(t, p) for t, p in self.scans if stamp - t <= self.keep_s]

    def level(self, pose):
        """All buffered points in the current level frame; z is height above the floor."""
        if not self.scans:
            return np.zeros((0, 3))
        return odom_to_level(np.concatenate([p for _, p in self.scans]), pose)

    def clear(self):
        self.scans = []


def obstacles(points_floor):
    """Points (level frame, z above floor) that could collide with the body."""
    z = points_floor[:, 2]
    return points_floor[(z > OBSTACLE_MIN_Z) & (z < OBSTACLE_MAX_Z)]


def _corridor(points_floor, ux, uy):
    """(along, across, extent, half) for a body-wide corridor along unit direction (ux, uy)."""
    along = points_floor[:, 0] * ux + points_floor[:, 1] * uy
    across = np.abs(-points_floor[:, 0] * uy + points_floor[:, 1] * ux)
    extent = abs(ux) * HALF_LENGTH + abs(uy) * HALF_WIDTH
    half = abs(ux) * HALF_WIDTH + abs(uy) * HALF_LENGTH + .05
    return along, across, extent, half


def obstacle_along(points_floor, ux, uy):
    """Free distance (m) from the body edge to the nearest obstacle along (ux, uy), within the corridor."""
    pts = obstacles(points_floor)
    if not len(pts):
        return math.inf
    along, across, extent, half = _corridor(pts, ux, uy)
    ahead = (along > 0) & (across < half)
    if not ahead.any():
        return math.inf
    return float(max(0., along[ahead].min() - extent))


def _filled(beyond, bin_m, max_m):
    """Distance covered by consecutive 10 cm steps that each hold at least one return, from the body edge."""
    bins = np.zeros(int(max_m / bin_m), bool)
    index = (beyond / bin_m).astype(int)
    bins[index[(index >= 0) & (index < len(bins))]] = True
    return float((np.argmin(bins) if not bins.all() else len(bins)) * bin_m)


def drop_along(points_floor, ux, uy, bin_m=.1, max_m=3.0):
    """Free distance (m) from the body edge to a drop along (ux, uy): where the supported floor ends, when the
    LiDAR sees returns more than STEP_DOWN_Z below the robot's floor in the corridor (a ledge, stairs going
    down, a hole). inf when nothing in the corridor is below the floor."""
    if not len(points_floor):
        return math.inf
    along, across, extent, half = _corridor(points_floor, ux, uy)
    inside = (across < half) & (along > extent)
    z = points_floor[:, 2]
    below = inside & (z < -STEP_DOWN_Z)
    if not below.any():
        return math.inf
    supported = inside & (z >= -STEP_DOWN_Z)            # floor (or anything standing on it) at our level
    floor_end = _filled(along[supported] - extent, bin_m, max_m)
    return float(max(0., min(floor_end, (along[below] - extent).min())))


def clearance_along(points_floor, ux, uy):
    """Free distance (m) from the body edge along unit direction (ux, uy), within a body-wide corridor: to the
    nearer of an obstacle and a drop."""
    return min(obstacle_along(points_floor, ux, uy), drop_along(points_floor, ux, uy))


def observed_along(points_floor, ux, uy, bin_m=.1, max_m=3.0):
    """How far beyond the body edge the corridor along (ux, uy) is actually observed (a return at the robot's
    floor level or above in every 10 cm step). Space the LiDAR cannot see is never assumed free, and a lower
    floor past a ledge does not count as seeing the corridor."""
    if not len(points_floor):
        return 0.
    along, across, extent, half = _corridor(points_floor, ux, uy)
    seen = (across < half) & (along > extent) & (points_floor[:, 2] >= -STEP_DOWN_Z)
    return _filled(along[seen] - extent, bin_m, max_m)


def turn_clearance(points_floor):
    """Free margin (m) around the body's turning circle: to the nearest obstacle, or to the nearest point
    below the floor (no floor under the circle means a ledge it could step off while turning)."""
    radius = math.hypot(HALF_LENGTH, HALF_WIDTH)
    margin = math.inf
    pts = obstacles(points_floor)
    if len(pts):
        margin = float(max(0., np.hypot(pts[:, 0], pts[:, 1]).min() - radius))
    if len(points_floor):
        below = points_floor[points_floor[:, 2] < -STEP_DOWN_Z]
        if len(below):
            margin = min(margin, float(max(0., np.hypot(below[:, 0], below[:, 1]).min() - radius)))
    return margin


def sector_summary(points_floor, max_range=5.0):
    """Nearest obstacle per 45° sector, measured from the body edge; None if clear within max_range."""
    pts = obstacles(points_floor)
    result = {name: None for name in SECTORS}
    if not len(pts):
        return result
    angle = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    dist = np.hypot(pts[:, 0], pts[:, 1])
    for i, name in enumerate(SECTORS):
        centre = i * 45
        delta = (angle - centre + 180) % 360 - 180
        sel = (np.abs(delta) <= 22.5) & (dist <= max_range)
        if sel.any():
            # Subtract the body extent in that direction so values mean "free space".
            ux, uy = math.cos(math.radians(centre)), math.sin(math.radians(centre))
            edge = abs(ux) * HALF_LENGTH + abs(uy) * HALF_WIDTH
            result[name] = round(max(0., float(dist[sel].min()) - edge), 2)
    return result


def sector_free_space(points_floor, max_range=5.0):
    """sector_summary, plus the distance to a drop in each direction (a ledge is reported like an obstacle,
    so `look` gives a number instead of "clear" on that side)."""
    result = sector_summary(points_floor, max_range)
    for i, name in enumerate(SECTORS):
        a = math.radians(i * 45)
        edge = drop_along(points_floor, math.cos(a), math.sin(a), max_m=min(max_range, 3.0))
        if edge != math.inf:
            edge = round(edge, 2)
            result[name] = edge if result[name] is None else min(result[name], edge)
    return result


COLOR_NAMES = [(0, "red"), (25, "orange"), (55, "yellow"), (110, "green"), (180, "cyan"), (225, "blue"),
               (275, "purple"), (320, "pink"), (360, "red")]
COLOR_WORDS = {name for _, name in COLOR_NAMES} | {"brown", "white", "grey", "gray", "black"}


def color_name(rgb_pixels):
    """Dominant colour word: the most common hue among saturated pixels, else a grey level."""
    if not len(rgb_pixels):
        return None
    rgb = rgb_pixels.reshape(-1, 3).astype(float) / 255.
    mx, mn = rgb.max(1), rgb.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    colourful = (sat > .25) & (mx > .2)
    if colourful.mean() < .35:
        v = float(np.median(mx))
        return "black" if v < .18 else "white" if v > .75 else "grey"
    r, g, b = rgb[colourful].T
    delta = np.maximum(mx[colourful] - mn[colourful], 1e-6)
    hue = np.where(mx[colourful] == r, ((g - b) / delta) % 6,
                   np.where(mx[colourful] == g, (b - r) / delta + 2, (r - g) / delta + 4)) * 60
    counts, edges = np.histogram(hue, bins=24, range=(0, 360))
    peak = int(counts.argmax())
    dominant = (edges[peak] + edges[peak + 1]) / 2
    in_peak = (hue >= edges[peak]) & (hue < edges[peak + 1])
    value = float(np.median(mx[colourful][in_peak]))
    if 15 <= dominant <= 45 and value < .65:
        return "brown"
    return min(COLOR_NAMES, key=lambda item: min(abs(dominant - item[0]), 360 - abs(dominant - item[0])))[1]


def _erode(mask, steps=2):
    m = mask.copy()
    for _ in range(steps):
        m[1:, :] &= mask[:-1, :]
        m[:-1, :] &= mask[1:, :]
        m[:, 1:] &= mask[:, :-1]
        m[:, :-1] &= mask[:, 1:]
        mask = m.copy()
    return m


def fuse(masks, rgb, cloud_base, intrinsics, camera_pos, camera_rot, roll, pitch, height):
    """Turn segmentation masks into detections with LiDAR range (or a ground-plane estimate).

    cloud_base: Nx3 points in the base frame. camera_rot maps OpenCV camera axes to base axes.
    """
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    h, w = rgb.shape[:2]
    cam = (cloud_base - camera_pos) @ camera_rot          # base -> camera optical frame
    front = cam[:, 2] > .1
    u = np.full(len(cam), -1)
    v = np.full(len(cam), -1)
    u[front] = np.round(fx * cam[front, 0] / cam[front, 2] + cx).astype(int)
    v[front] = np.round(fy * cam[front, 1] / cam[front, 2] + cy).astype(int)
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    level_cloud = base_to_level(cloud_base, roll, pitch)
    level_rot = rotation_rp(roll, pitch)
    detections = []
    for n, item in enumerate(sorted(masks, key=lambda m: -m.mask.sum())):
        core = _erode(item.mask) if item.mask.sum() > 60 else item.mask
        hits = inside.copy()
        hits[inside] = core[v[inside], u[inside]]
        rows, cols = np.nonzero(item.mask)
        centre_dir = np.array([(cols.mean() - cx) / fx, (rows.mean() - cy) / fy, 1.])
        level_dir = level_rot @ (camera_rot @ centre_dir)
        bearing = math.degrees(math.atan2(level_dir[1], level_dir[0]))
        x = y = distance = None
        source = "unknown"
        pts = level_cloud[hits]
        if len(pts):
            horizontal = np.hypot(pts[:, 0], pts[:, 1])
            near = pts[horizontal <= np.percentile(horizontal, 20) + .25]
            x, y = float(np.median(near[:, 0])), float(np.median(near[:, 1]))
            distance, source = math.hypot(x, y), "lidar"
            bearing = math.degrees(math.atan2(y, x))   # from the body centre, not the camera
        else:
            # Monocular fallback: the lowest mask pixel is where the object meets the floor.
            bottom = rows.max()
            col = cols[rows == bottom].mean()
            ray = level_rot @ (camera_rot @ np.array([(col - cx) / fx, (bottom - cy) / fy, 1.]))
            origin = level_rot @ camera_pos
            cam_height = height + origin[2]
            if ray[2] < -1e-3:
                t = cam_height / -ray[2]
                gx, gy = float(origin[0] + t * ray[0]), float(origin[1] + t * ray[1])
                if math.hypot(gx, gy) <= GROUND_PLANE_MAX_M:
                    x, y, distance, source = gx, gy, math.hypot(gx, gy), "ground_plane"
                bearing = math.degrees(math.atan2(gy, gx))
        clothing = None
        if item.label == "person":
            top, bottom = rows.min(), rows.max()
            torso = (rows >= top + .2 * (bottom - top)) & (rows <= top + .55 * (bottom - top))
            clothing = color_name(rgb[rows[torso], cols[torso]])
        h, w = item.mask.shape
        box = [round(cols.min() / w, 4), round(rows.min() / h, 4), round((cols.max() + 1) / w, 4),
               round((rows.max() + 1) / h, 4)]
        detections.append(Detection(
            id=f"obj{n + 1}", label=item.label, color=None if item.label in NO_COLOR else color_name(rgb[item.mask]),
            distance_m=distance, bearing_deg=bearing, x_m=x, y_m=y, distance_source=source,
            confidence=float(item.score), clothing=clothing, box=box))
    return sorted(detections, key=lambda d: (d.distance_m is None, d.distance_m or 0))


def describe_bearing(bearing):
    a = abs(bearing)
    if a < 8:
        return "straight ahead"
    side = "left" if bearing > 0 else "right"
    if a < 30:
        return f"slightly {side}"
    if a < 70:
        return f"ahead-{side}"
    return f"to the {side}"
