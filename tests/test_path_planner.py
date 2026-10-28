
import unittest
import numpy as np

from go2.path_planner import astar, simplify


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

    def test_does_not_travel_through_unknown_cells(self):      # from Prionti's handover branch
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = -1
        self.assertIsNone(astar(grid, (0, 0), (4, 4)))

    def test_unknown_cells_can_be_allowed_at_a_cost(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = -1
        self.assertIsNotNone(astar(grid, (0, 0), (4, 4), unknown_cost=3))

    def test_routes_are_the_shortest(self):
        # The cost used to be the heap's tie-break counter, so some routes came out longer than the shortest.
        rng = np.random.default_rng(7)
        for _ in range(150):
            grid = (rng.random((20, 20)) < .3).astype(int)
            grid[0, 0] = grid[19, 19] = 0
            shortest = bfs_length(grid, (0, 0), (19, 19))
            path = astar(grid, (0, 0), (19, 19))
            self.assertEqual(None if path is None else len(path) - 1, shortest)

    def test_blocked_cells_are_avoided(self):
        grid = np.zeros((5, 5), dtype=int)
        blocked = np.zeros((5, 5), dtype=bool)
        blocked[1, :4] = True
        path = astar(grid, (0, 0), (4, 0), blocked=blocked)
        self.assertTrue(all(not blocked[c] for c in path))

    def test_diagonal_routes_dont_cut_corners(self):
        grid = np.zeros((3, 3), dtype=int)
        grid[0, 1] = grid[1, 0] = 1
        self.assertIsNone(astar(grid, (0, 0), (1, 1), diagonal=True))

    def test_simplify_keeps_only_the_corners(self):
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        clear = lambda a, b: a[0] == b[0] or a[1] == b[1]      # only straight lines along rows or columns
        self.assertEqual(simplify(path, clear), [(0, 0), (0, 2), (2, 2)])


def bfs_length(grid, start, goal):
    seen, frontier, steps = {start}, [start], 0
    while frontier:
        if goal in frontier:
            return steps
        nxt = []
        for r, c in frontier:
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1] and grid[n] == 0 and n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier, steps = nxt, steps + 1
    return None


if __name__ == "__main__":
    unittest.main()
