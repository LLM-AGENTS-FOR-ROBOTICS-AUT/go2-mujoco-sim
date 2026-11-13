"""Bounded, sensor-only SE(2) scan-to-map localisation.

The interface consumes a raw odometry pose, a levelled floor-relative LiDAR
cloud and its sensor timestamp. It knows nothing about MuJoCo or scene truth;
a hardware adapter can supply the same inputs after hardware validation (#34).
Reference points enter the map only after an accepted match. Sparse, degenerate
or inconsistent scans are rejected rather than drawn as a second set of walls.
"""
import math
import threading
from collections import deque
import numpy as np


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def transform(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]


def compose(a, b):
    return np.array([*transform(np.asarray([b[:2]]), a)[0], wrap(a[2]+b[2])])


def inverse(pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([-c*pose[0]-s*pose[1], s*pose[0]-c*pose[1], -pose[2]])


def _voxel(points, resolution=.08, limit=600):
    if not len(points):
        return points
    _, index = np.unique(np.floor(points/resolution).astype(np.int64), axis=0, return_index=True)
    result = points[np.sort(index)]
    if len(result) > limit:
        result = result[np.linspace(0, len(result)-1, limit, dtype=int)]
    return result


def _nearest(points, reference):
    indexes, distances = [], []
    for offset in range(0, len(points), 96):
        d = ((points[offset:offset+96, None]-reference[None])**2).sum(axis=2)
        i = d.argmin(axis=1)
        indexes.extend(i)
        distances.extend(d[np.arange(len(i)), i])
    return np.asarray(indexes), np.sqrt(distances)


def _planes(points, reference):
    """Local surface normals from five nearby map returns (no scene geometry)."""
    centres, normals, distances, linear = [], [], [], []
    for offset in range(0, len(points), 96):
        query = points[offset:offset+96]
        d = ((query[:, None]-reference[None])**2).sum(axis=2)
        index = np.argpartition(d, 4, axis=1)[:, :5]
        neighbours = reference[index]
        centre = neighbours.mean(axis=1)
        delta = neighbours-centre[:, None]
        eigen, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', delta, delta)/5.)
        centres.extend(centre)
        normals.extend(vectors[:, :, 0])
        distances.extend(np.sqrt(d.min(axis=1)))
        linear.extend((eigen[:, 0] < .15*np.maximum(eigen[:, 1], 1e-9)) &
                      (np.take_along_axis(d, index, axis=1).max(axis=1) < .3**2))
    return np.asarray(centres), np.asarray(normals), np.asarray(distances), np.asarray(linear)


class ScanLocalizer:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.map_from_odom = np.zeros(3)
            self.reference = {}
            self.last_stamp = None
            self.last_added = None
            self.failures = 0
            self.detail = {"state": "initialising", "accepted": False, "matches": 0}
            self.recent = deque(maxlen=4)

    def pose(self, odom):
        with self.lock:
            return compose(self.map_from_odom, odom)

    def status(self):
        with self.lock:
            return dict(self.detail, failures=self.failures, reference_points=len(self.reference),
                        scan_stamp=self.last_stamp)

    def _reject(self, reason):
        self.failures += 1
        self.detail = {"state": "lost" if self.failures >= 3 else "degraded",
                       "accepted": False, "reason": reason, "matches": 0}

    def update(self, odom, floor_cloud, stamp):
        odom = np.asarray(odom, dtype=float)
        cloud = np.asarray(floor_cloud, dtype=float)
        if odom.shape != (3,) or not np.isfinite(odom).all() or not math.isfinite(stamp):
            raise ValueError("localisation needs a finite pose and sensor timestamp")
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError("localisation needs an Nx3 cloud")
        with self.lock:
            predicted = compose(self.map_from_odom, odom)
            if self.last_stamp is not None and stamp <= self.last_stamp:
                return predicted
            self.last_stamp = stamp
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            distance = np.linalg.norm(cloud[:, :2], axis=1)
            scan = cloud[(cloud[:, 2] > .15) & (cloud[:, 2] < 1.5) &
                         (distance > .65) & (distance < 7.), :2]
            if len(scan) >= 20:
                # The L1 pattern is non-repetitive: accumulate a short window
                # in raw odometry, accounting for motion between captures.
                # No corrected pose or truth enters this alignment window.
                self.recent.append((stamp, transform(scan, odom)))
                recent = [points for when, points in self.recent if stamp-when <= .6]
                scan = transform(np.vstack(recent), inverse(odom))
            scan = _voxel(scan)
            if len(scan) < 40:
                self._reject("too few raised LiDAR returns")
                return predicted
            if not self.reference:
                eigen = np.linalg.eigvalsh(np.cov(scan.T))
                if eigen[0] < .01 or eigen[0]/max(eigen[1], 1e-6) < .015:
                    self._reject("initial geometry cannot constrain both position axes")
                    return predicted
                self._add(transform(scan, predicted))
                self.last_added = predicted.copy()
                self.failures = 0
                self.detail = {"state": "tracking", "accepted": True, "matches": len(scan), "rms_m": 0.}
                return predicted
            reference = np.asarray(list(self.reference.values()))
            reference = reference[np.linalg.norm(reference-predicted[:2], axis=1) < 7.5]
            reference = _voxel(reference, resolution=.07, limit=5000)
            if len(reference) < 40:
                self._reject("no overlapping reference map")
                return predicted
            pose = predicted.copy()
            for _ in range(15):
                world = transform(scan, pose)
                index, distance = _nearest(world, reference)
                mask = distance < .35
                if mask.sum() < max(35, .35*len(scan)):
                    self._reject("insufficient scan overlap")
                    return predicted
                # Trim dynamic returns and unmatched newly exposed surfaces.
                threshold = max(.025, float(np.quantile(distance[mask], .85)))
                mask &= distance <= threshold
                a, b = world[mask], reference[index[mask]]
                ac, bc = a.mean(axis=0), b.mean(axis=0)
                u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
                rotation = vt.T @ u.T
                if np.linalg.det(rotation) < 0:
                    vt[-1] *= -1
                    rotation = vt.T @ u.T
                angle = math.atan2(rotation[1, 0], rotation[0, 0])
                shift = bc-rotation @ ac
                pose = compose([*shift, angle], pose)
                if np.linalg.norm(shift) < .0005 and abs(angle) < .0002:
                    break
            # Point-to-plane refinement avoids sliding between sparse LiDAR
            # samples on a wall. Fit normals from map neighbours, never truth.
            for _ in range(6):
                world = transform(scan, pose)
                centres, normals, distance, linear = _planes(world, reference)
                residual = ((centres-world)*normals).sum(axis=1)
                mask = linear & (distance < .2) & (abs(residual) < .12)
                if mask.sum() < 30:
                    break
                n = normals[mask]
                relative = world[mask]-pose[:2]
                matrix = np.column_stack([n, -n[:, 0]*relative[:, 1]+n[:, 1]*relative[:, 0]])
                step, _, rank, _ = np.linalg.lstsq(matrix, residual[mask], rcond=None)
                if rank < 3 or np.linalg.norm(step[:2]) > .15 or abs(step[2]) > .1:
                    break
                pose[:2] += step[:2]
                pose[2] = wrap(pose[2]+step[2])
                if np.linalg.norm(step[:2]) < .0005 and abs(step[2]) < .0002:
                    break
            world = transform(scan, pose)
            index, distance = _nearest(world, reference)
            mask = distance < .12
            if mask.sum() < max(35, .4*len(scan)):
                self._reject("scan does not fit the reference map")
                return predicted
            eigen = np.linalg.eigvalsh(np.cov(reference[index[mask]].T))
            rms = float(np.sqrt(np.mean(distance[mask]**2)))
            if eigen[0] < .01 or eigen[0]/max(eigen[1], 1e-6) < .015:
                self._reject("geometry cannot constrain both position axes")
                return predicted
            if rms > .075 or np.linalg.norm(pose[:2]-predicted[:2]) > .3 or abs(wrap(pose[2]-predicted[2])) > .15:
                self._reject("scan correction exceeds confidence bounds")
                return predicted
            self.map_from_odom = compose(pose, inverse(odom))
            self.failures = 0
            self.detail = {"state": "tracking", "accepted": True, "matches": int(mask.sum()), "rms_m": rms}
            if (np.linalg.norm(pose[:2]-self.last_added[:2]) > .15 or
                    abs(wrap(pose[2]-self.last_added[2])) > .1 or len(self.reference) < 1000):
                # Extend into new space only when the scan itself has strong
                # overlap with the existing map; failed scans never enter it.
                centres, normals, distance, linear = _planes(world, reference)
                stable = linear & (distance < .15)
                world[stable] += (((centres[stable]-world[stable])*normals[stable]).sum(axis=1)[:, None]
                                  * normals[stable])
                self._add(world)
                self.last_added = pose.copy()
            return pose

    def _add(self, points):
        for point in points:
            key = tuple(np.floor(point/.06).astype(int))
            self.reference.setdefault(key, point.copy())
        # Bound storage while retaining old anchors for revisits.
        if len(self.reference) > 25000:
            keys = list(self.reference)
            for key in keys[1::2]:
                self.reference.pop(key, None)
