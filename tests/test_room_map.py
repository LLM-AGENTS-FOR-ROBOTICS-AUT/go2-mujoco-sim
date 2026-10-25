
import unittest

import numpy as np

from go2.room_map import OccupancyGrid


class OccupancyGridTests(unittest.TestCase):
    def test_grid_starts_unknown(self):
        room_map = OccupancyGrid()
        self.assertEqual(room_map.grid.shape, (100, 100))
        self.assertTrue(np.all(room_map.grid == -1))

    def test_marks_obstacle(self):
        room_map = OccupancyGrid()
        room_map.mark_obstacles(np.array([[1.0, 0.0]]))

        cell = room_map.world_to_cell(1.0, 0.0)
        self.assertEqual(room_map.grid[cell], 1)

    def test_ignores_points_outside_grid(self):
        room_map = OccupancyGrid()
        room_map.mark_obstacles(np.array([[100.0, 100.0]]))

        self.assertTrue(np.all(room_map.grid == -1))

    def test_rejects_invalid_point_shape(self):
        room_map = OccupancyGrid()

        with self.assertRaises(ValueError):
            room_map.mark_obstacles(np.array([1.0, 2.0]))

    def test_mark_ray_marks_endpoint_occupied(self):
        room_map = OccupancyGrid()

        room_map.mark_ray((0.0, 0.0), (1.0, 0.0))

        end_cell = room_map.world_to_cell(1.0, 0.0)
        self.assertEqual(room_map.grid[end_cell], 1)

    def test_mark_ray_marks_cells_before_endpoint_free(self):
        room_map = OccupancyGrid()

        room_map.mark_ray((0.0, 0.0), (1.0, 0.0))

        start_cell = room_map.world_to_cell(0.0, 0.0)
        self.assertEqual(room_map.grid[start_cell], 0)

    def test_mark_ray_ignores_endpoint_outside_grid(self):
        room_map = OccupancyGrid()

        room_map.mark_ray((0.0, 0.0), (100.0, 100.0))

        self.assertEqual(np.count_nonzero(room_map.grid == 1), 0)


if __name__ == "__main__":
    unittest.main()
