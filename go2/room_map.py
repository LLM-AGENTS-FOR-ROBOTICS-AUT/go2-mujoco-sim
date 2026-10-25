
"""A simple 2D occupancy grid for robot room mapping."""

import math

import numpy as np


class OccupancyGrid:
    """Store observed obstacles in a 2D grid."""

    def __init__(self, resolution=0.1, size=100):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if size <= 0:
            raise ValueError("size must be positive")

        self.resolution = float(resolution)
        self.size = int(size)
        self.grid = np.full(
            (self.size, self.size), -1, dtype=np.int8
        )

    def world_to_cell(self, x, y):
        """Convert world coordinates in metres to grid indices."""
        col = math.floor(x / self.resolution) + self.size // 2
        row = math.floor(y / self.resolution) + self.size // 2

        if not (0 <= row < self.size and 0 <= col < self.size):
            return None

        return row, col

    def cell_to_world(self, row, col):
        """Convert grid indices to world coordinates in metres."""
        x = (col - self.size // 2 + 0.5) * self.resolution
        y = (row - self.size // 2 + 0.5) * self.resolution
        return x, y 

    def mark_obstacles(self, points):
        """Mark observed obstacle points as occupied.

        points should be an Nx2 NumPy array of x,y coordinates in metres,
        expressed in the same world coordinate frame as the grid.
        """
        points = np.asarray(points, dtype=float)

        if points.size == 0:
            return

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        for x, y in points:
            cell = self.world_to_cell(x, y)
            if cell is not None:
                self.grid[cell] = 1

    
    def mark_ray(self, start, end):
        """Mark free cells along a ray and its endpoint as occupied."""
        start_cell = self.world_to_cell(*start)
        end_cell = self.world_to_cell(*end)

        if start_cell is None or end_cell is None:
            return

        row0, col0 = start_cell
        row1, col1 = end_cell

        steps = max(abs(row1 - row0), abs(col1 - col0))

        if steps == 0:
            self.grid[end_cell] = 1
            return

        rows = np.rint(
            np.linspace(row0, row1, steps + 1)
        ).astype(int)
        cols = np.rint(
            np.linspace(col0, col1, steps + 1)
        ).astype(int)

        # Cells along the ray are free.
        self.grid[rows[:-1], cols[:-1]] = 0

        # The endpoint is occupied.
        self.grid[end_cell] = 1

