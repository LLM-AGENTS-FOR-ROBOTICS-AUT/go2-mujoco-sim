"""A 2D occupancy grid of the room, built from the LiDAR (#14).

Cells are -1 unknown, 0 free, 1 occupied, in odometry coordinates (the frame the robot's x/y are reported in).
Each scan marks the cells between the robot and every return as free, then the returns that could block the
body as occupied. Marking free space every scan also clears things that have moved, such as a person walking
past.

Limits, worth revisiting: the grid is a fixed square around the odometry origin (20 m x 20 m by default), so
points beyond it are dropped; and drops (ledges, stairs down) aren't marked yet, so a planned route can lead to a
ledge, where the executor's own drop check (#22) stops the robot."""

import math

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 1
FLOOR_Z = .06              # a return this close to floor height is floor, so its cell is free
OBSTACLE_Z = (.06, .55)    # heights that can block the body (perception.OBSTACLE_MIN_Z/MAX_Z)
RAY_RANGE_M = 4.           # free space is only marked this far out; further returns are too sparse to trust
MAX_RAYS = 600             # rays used per scan (a random subset; the scans in between fill in the rest)


class OccupancyGrid:
    """Store what the LiDAR has seen in a 2D grid."""

    def __init__(self, resolution=0.1, size=200):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if size <= 0:
            raise ValueError("size must be positive")
        self.resolution = float(resolution)
        self.size = int(size)
        self.grid = np.full((self.size, self.size), UNKNOWN, dtype=np.int8)
        self._rng = np.random.default_rng(0)

    # ---- coordinates ---------------------------------------------------------------------------
    def world_to_cell(self, x, y):
        """Grid (row, col) of a point in metres, or None outside the grid."""
        col = math.floor(x / self.resolution) + self.size // 2
        row = math.floor(y / self.resolution) + self.size // 2
        if not (0 <= row < self.size and 0 <= col < self.size):
            return None
        return row, col

    def cell_to_world(self, row, col):
        """Centre of a cell in metres."""
        x = (col - self.size // 2 + 0.5) * self.resolution
        y = (row - self.size // 2 + 0.5) * self.resolution
        return x, y

    def _cells(self, xy):
        """(rows, cols, inside) for an Nx2 array of points, all at once."""
        cols = np.floor(xy[:, 0] / self.resolution).astype(int) + self.size // 2
        rows = np.floor(xy[:, 1] / self.resolution).astype(int) + self.size // 2
        inside = (rows >= 0) & (rows < self.size) & (cols >= 0) & (cols < self.size)
        return rows, cols, inside

    # ---- marking -------------------------------------------------------------------------------
    def mark_obstacles(self, points):
        """Mark points (Nx2, metres, odometry frame) as occupied."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        rows, cols, inside = self._cells(points)
        self.grid[rows[inside], cols[inside]] = OCCUPIED

    def mark_ray(self, start, end, endpoint_occupied=True):
        """Mark the cells from start to end as free, and the end cell as occupied (a single return) unless
        endpoint_occupied is False (then the whole ray is free)."""
        start_cell, end_cell = self.world_to_cell(*start), self.world_to_cell(*end)
        if start_cell is None or end_cell is None:
            return
        (row0, col0), (row1, col1) = start_cell, end_cell
        steps = max(abs(row1 - row0), abs(col1 - col0))
        if steps:
            rows = np.rint(np.linspace(row0, row1, steps + 1)).astype(int)
            cols = np.rint(np.linspace(col0, col1, steps + 1)).astype(int)
            self.grid[rows[:-1], cols[:-1]] = FREE
        self.grid[end_cell] = OCCUPIED if endpoint_occupied else FREE

    def mark_scan(self, origin, points):
        """Add one LiDAR scan: origin is the robot's (x, y); points are Nx3 in the odometry frame with z the
        height above the floor. Cells along each ray are free; floor returns are free; returns at heights that
        can block the body are occupied."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return
        z = points[:, 2]
        ox, oy = origin
        d = np.hypot(points[:, 0] - ox, points[:, 1] - oy)
        floor = (np.abs(z) < FLOOR_Z) & (d <= RAY_RANGE_M)
        blocking = (z > OBSTACLE_Z[0]) & (z < OBSTACLE_Z[1])
        rays = points[(floor | blocking) & (d <= RAY_RANGE_M) & (d > 1e-6)]
        if len(rays) > MAX_RAYS:
            rays = rays[self._rng.choice(len(rays), MAX_RAYS, replace=False)]
        if len(rays):
            length = np.hypot(rays[:, 0] - ox, rays[:, 1] - oy)
            steps = np.maximum(1, np.ceil(length / self.resolution).astype(int))
            n = int(steps.max())
            t = np.arange(n) / n                                             # 0 .. just short of the return
            frac = np.minimum(t[None, :] * n / steps[:, None], 1 - 1e-9)     # each ray stops before its end
            xs = ox + (rays[:, 0:1] - ox) * frac
            ys = oy + (rays[:, 1:2] - oy) * frac
            rows, cols, inside = self._cells(np.stack([xs.ravel(), ys.ravel()], 1))
            self.grid[rows[inside], cols[inside]] = FREE
        floor_pts = points[floor]
        if len(floor_pts):
            rows, cols, inside = self._cells(floor_pts[:, :2])
            keep = self.grid[rows[inside], cols[inside]] != OCCUPIED
            self.grid[rows[inside][keep], cols[inside][keep]] = FREE
        self.mark_obstacles(points[blocking][:, :2])

    # ---- planning helpers ----------------------------------------------------------------------
    def blocked(self, radius_m):
        """Boolean grid of cells the body's centre can't be in: occupied cells grown by radius_m."""
        occupied = self.grid == OCCUPIED
        reach = int(math.ceil(radius_m / self.resolution))
        if reach <= 0 or not occupied.any():
            return occupied
        out = occupied.copy()
        rows, cols = np.nonzero(occupied)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                if dr * dr + dc * dc > reach * reach:
                    continue
                r, c = rows + dr, cols + dc
                ok = (r >= 0) & (r < self.size) & (c >= 0) & (c < self.size)
                out[r[ok], c[ok]] = True
        return out

    def line_clear(self, blocked, a, b):
        """True if the straight line between cells a and b crosses no blocked cell."""
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if steps == 0:
            return not blocked[a]
        rows = np.rint(np.linspace(a[0], b[0], steps * 2 + 1)).astype(int)
        cols = np.rint(np.linspace(a[1], b[1], steps * 2 + 1)).astype(int)
        return not blocked[rows, cols].any()
