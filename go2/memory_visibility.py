"""Evidence that a remembered location is clear, independent of detector misses."""
import math

import numpy as np
from . import perception as P


def clear_saved_location(item, frame, floor_cloud):
    """Conservative flat-scene check, used only in the simulator for now.

    Require a stationary fresh camera view, an interior image projection, direct
    floor returns around the saved point, and no raised returns there or on the
    line of sight. Occlusion, blind spots and sparse returns leave memory alone.
    """
    state = frame.state
    if state.age > .5 or state.speed > .05 or abs(state.wz) > .1 or floor_cloud is None:
        return False
    if state.posture != "standing" or max(abs(state.roll), abs(state.pitch)) > .3:
        return False
    localization = state.extra.get("localization")
    if localization is not None and not localization.get("accepted"):
        return False
    cloud = np.asarray(floor_cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        return False
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if not len(cloud):
        return False
    pose = state.x, state.y, state.yaw
    points = np.array([[item["x"], item["y"], .05-state.height],
                       [item["x"], item["y"], .5-state.height]])
    level = P.odom_to_level(points, pose)
    distance = math.hypot(level[0, 0], level[0, 1])
    if not .5 <= distance <= 2.5:
        return False
    base = level @ P.rotation_rp(state.roll, state.pitch)
    camera = (base-frame.camera_pos) @ frame.camera_rot
    if np.any(camera[:, 2] <= .2):
        return False
    k = frame.intrinsics
    uv = camera[:, :2]/camera[:, 2:3] * [k["fx"], k["fy"]] + [k["cx"], k["cy"]]
    height, width = frame.rgb.shape[:2]
    if not ((uv[:, 0] > .1*width).all() and (uv[:, 0] < .9*width).all()
            and (uv[:, 1] > .1*height).all() and (uv[:, 1] < .9*height).all()):
        return False
    xy = level[0, :2]
    near = np.linalg.norm(cloud[:, :2]-xy, axis=1) < .15
    floor = np.abs(cloud[:, 2]) <= .02
    raised = (cloud[:, 2] > .025) & (cloud[:, 2] < 1.5)
    if np.count_nonzero(near & floor) < 6 or np.any(near & raised):
        return False
    along = cloud[:, :2] @ (xy/distance)
    across = abs(cloud[:, 0]*xy[1]/distance-cloud[:, 1]*xy[0]/distance)
    occluding = raised & (along > .3) & (along < distance-.2) & (across < .15)
    return not np.any(occluding)
