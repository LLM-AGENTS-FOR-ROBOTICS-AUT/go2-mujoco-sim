
import unittest
import numpy as np

from go2.path_planner import astar


class TestAStar(unittest.TestCase):

    def test_finds_path_on_clear_grid(self):
        grid = np.zeros((5, 5), dtype=int)

        path = astar(grid, (0, 0), (4, 4))

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

    def test_routes_around_obstacle(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[1, :] = 1
        grid[1, 2] = 0  # Gap in the obstacle wall

        path = astar(grid, (0, 0), (4, 4))

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

        # Every step should be adjacent, and no step should be occupied.
        for row, col in path:
            self.assertEqual(grid[row, col], 0)

        for current, following in zip(path, path[1:]):
            distance = (
                abs(current[0] - following[0])
                + abs(current[1] - following[1])
            )
            self.assertEqual(distance, 1)

    def test_returns_none_when_goal_is_blocked(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[4, 4] = 1

        self.assertIsNone(astar(grid, (0, 0), (4, 4)))

    def test_returns_none_when_no_route_exists(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = 1

        self.assertIsNone(astar(grid, (0, 0), (4, 4)))


if __name__ == "__main__":
    unittest.main()
