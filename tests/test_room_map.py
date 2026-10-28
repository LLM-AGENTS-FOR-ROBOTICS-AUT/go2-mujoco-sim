import math
import unittest

import numpy as np

from go2.executor import Executor
from go2.room_map import OccupancyGrid
from tests.fakes import FakeRobot


class OccupancyGridTests(unittest.TestCase):
    def test_grid_starts_unknown(self):
        room_map = OccupancyGrid()
        self.assertEqual(room_map.grid.shape, (200, 200))   # 20 m x 20 m at 0.1 m
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


    def test_mark_ray_can_leave_the_endpoint_free(self):
        room_map = OccupancyGrid()
        room_map.mark_ray((0.0, 0.0), (1.0, 0.0), endpoint_occupied=False)
        self.assertEqual(room_map.grid[room_map.world_to_cell(1.0, 0.0)], 0)

    def test_a_scan_marks_free_space_and_obstacles(self):
        room_map = OccupancyGrid()
        floor = np.array([[x, y, 0.] for x in np.arange(-2, 2, .05) for y in np.arange(-2, 2, .05)])
        box = np.array([[1.5, y, .3] for y in np.arange(-.3, .3, .05)])
        room_map.mark_scan((0., 0.), np.vstack([floor, box]))
        self.assertEqual(room_map.grid[room_map.world_to_cell(1.5, 0.)], 1)
        self.assertEqual(room_map.grid[room_map.world_to_cell(.8, 0.)], 0)
        self.assertEqual(room_map.grid[room_map.world_to_cell(-1., -1.)], 0)
        self.assertEqual(room_map.grid[room_map.world_to_cell(3., 3.)], -1, "never seen")

    def test_a_scan_clears_something_that_moved_away(self):
        room_map = OccupancyGrid()
        room_map.mark_obstacles(np.array([[1.0, 0.0]]))                   # a person who has since walked off
        room_map.mark_scan((0., 0.), np.array([[2.0, 0., 0.]]))           # now the LiDAR sees floor behind
        self.assertEqual(room_map.grid[room_map.world_to_cell(1.0, 0.0)], 0)

    def test_floor_below_the_robots_is_marked_as_a_drop(self):
        room_map = OccupancyGrid()
        room_map.mark_scan((0., 0.), np.array([[.8, 0., 0.], [1.6, 0., -.5]]))    # platform, then the floor below
        self.assertEqual(room_map.grid[room_map.world_to_cell(.8, 0.)], 0)
        self.assertEqual(room_map.grid[room_map.world_to_cell(1.6, 0.)], 1)

    def test_blocked_grows_obstacles_by_the_body_size(self):
        room_map = OccupancyGrid()
        room_map.mark_obstacles(np.array([[1.0, 0.0]]))
        blocked = room_map.blocked(.3)
        self.assertTrue(blocked[room_map.world_to_cell(1.25, 0.0)])
        self.assertFalse(blocked[room_map.world_to_cell(1.5, 0.0)])


class RobotMapTests(unittest.TestCase):
    """The executor's map, from the FakeRobot's LiDAR."""

    def setUp(self):
        self.robot = FakeRobot(obstacles=[(1.5, 1.8, -.5, .5)])     # one box 1.5 m in front, fixed in the room
        self.robot.connect()
        self.executor = Executor()
        self.executor.attach(self.robot)
        self.addCleanup(self.robot.close)
        self.addCleanup(self.executor.close)

    def box_cells(self):
        m = self.executor.room_map
        m.grid[:] = -1
        self.executor.cloud.clear()
        self.executor._refresh_cloud()
        rows, cols = np.nonzero(m.grid == 1)
        xs, ys = zip(*(m.cell_to_world(r, c) for r, c in zip(rows, cols)))
        return np.mean(xs), np.mean(ys)

    def test_obstacles_stay_put_when_the_robot_turns(self):
        before = self.box_cells()
        self.robot.yaw = math.pi / 2                       # turn left on the spot; the box hasn't moved
        after = self.box_cells()
        self.assertAlmostEqual(before[0], 1.65, delta=.15)
        self.assertAlmostEqual(after[0], before[0], delta=.1)
        self.assertAlmostEqual(after[1], before[1], delta=.1)

    def test_free_space_is_marked(self):
        self.box_cells()
        m = self.executor.room_map
        self.assertGreater(int((m.grid == 0).sum()), 500)
        self.assertEqual(m.grid[m.world_to_cell(.8, 0.)], 0)


if __name__ == "__main__":
    unittest.main()
