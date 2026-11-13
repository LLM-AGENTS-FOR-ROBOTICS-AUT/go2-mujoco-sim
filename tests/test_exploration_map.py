import unittest

import numpy as np

from go2.exploration import ExplorationMap, next_viewpoint
from go2.room_map import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from go2.room_regions import RoomRegions


class GrowingMapTests(unittest.TestCase):
    def test_grows_in_every_direction_without_changing_world_positions(self):
        m = OccupancyGrid(size=20, grow=True)
        m.mark_obstacles([[.1, .2]])
        for point in ([12., 1.], [-14., -20.], [25., 25.]):
            m.mark_obstacles([point])
            self.assertEqual(m.grid[m.world_to_cell(.1, .2)], OCCUPIED)
            self.assertEqual(m.grid[m.world_to_cell(*point)], OCCUPIED)
            centre = m.cell_to_world(*m.world_to_cell(*point))
            self.assertLess(np.linalg.norm(np.array(centre) - point), .15)

    def test_snapshot_preserves_grown_origin_and_is_independent(self):
        m = OccupancyGrid(size=20, grow=True)
        m.mark_ray((0, 0), (-12, 0))
        snapshot = m.snapshot()
        self.assertEqual(snapshot.world_to_cell(-12, 0), m.world_to_cell(-12, 0))
        self.assertEqual(snapshot.cell_to_world(0, 0), m.cell_to_world(0, 0))
        snapshot.grid[:] = UNKNOWN
        self.assertEqual(m.grid[m.world_to_cell(-12, 0)], OCCUPIED)

    def test_growth_limit_refuses_an_outlier_without_changing_map(self):
        m = OccupancyGrid(size=20, grow=True, max_size=100)
        before = m.grid.copy()
        with self.assertRaises(ValueError):
            m.mark_obstacles([[10000, 0]])
        np.testing.assert_array_equal(m.grid, before)
        self.assertEqual(m.size, 20)

    def test_scan_grows_and_marks_outside_original_bounds(self):
        m = OccupancyGrid(size=20, grow=True)
        m.mark_scan((15., -17.), np.array([[16., -17., .3], [15.5, -17., 0.]]))
        self.assertEqual(m.grid[m.world_to_cell(16., -17.)], OCCUPIED)
        self.assertEqual(m.grid[m.world_to_cell(15.5, -17.)], FREE)

    def test_nonfinite_returns_are_ignored(self):
        m = OccupancyGrid(grow=True)
        m.mark_scan((0, 0), [[np.nan, 0, .3], [1, 0, .3]])
        self.assertEqual(m.grid[m.world_to_cell(1, 0)], OCCUPIED)

    def test_frontier_selector_uses_grown_origin(self):
        m = OccupancyGrid(size=20, grow=True)
        m.ensure_bounds([[20, 20]])
        m.grid[:] = OCCUPIED
        row, col = m.world_to_cell(20, 20)
        m.grid[row-8:row+1, col-15:col-4] = FREE
        m.grid[row-8:row+1, col-4:col] = UNKNOWN
        pose = (*m.cell_to_world(row-4, col-10), 0.)
        result = next_viewpoint(ExplorationMap(m, pose, clearance_m=.2), [])
        self.assertIsNotNone(result)
        self.assertGreater(result[0], 15.)
        self.assertGreater(result[1], 15.)


class LowObstacleTests(unittest.TestCase):
    def test_projected_ray_does_not_erase_an_obstacle_it_can_pass_above(self):
        for height in (.1, .2):
            with self.subTest(height=height):
                m = OccupancyGrid(resolution=.05, detect_low=True)
                m.mark_scan((0, 0), [[1.02, .02, height]])
                for _ in range(10):
                    m.mark_scan((0, 0), [[2.02, .02, 0.]])
                self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], OCCUPIED)

    def test_direct_floor_evidence_clears_a_removed_structural_obstacle(self):
        m = OccupancyGrid(resolution=.05, detect_low=True)
        m.mark_scan((0, 0), [[1.02, .02, .2]])
        m.mark_scan((0, 0), [[1.02, .02, 0.]])
        self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], OCCUPIED)
        for _ in range(12):
            m.mark_scan((0, 0), [[1.02, .02, 0.]])
        self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], FREE)

    def test_current_structural_hit_wins_over_prior_floor_evidence(self):
        m = OccupancyGrid(resolution=.05, detect_low=True)
        m.mark_scan((0, 0), [[1.02, .02, 0.]] * 32)
        m.mark_scan((0, 0), [[1.02, .02, 0.], [1.02, .02, .2]])
        self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], OCCUPIED)
        m.mark_scan((0, 0), [[1.02, .02, 2.]])
        self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], OCCUPIED)
        self.assertNotEqual(m.grid[m.world_to_cell(1.5, .5)], OCCUPIED)

    def test_repeated_book_height_is_an_obstacle_and_uncertainty_is_not_free(self):
        m = OccupancyGrid(resolution=.05, detect_low=True)
        returns = [[1.02, .02, .04], [1.02, .02, .04]]
        m.mark_scan((0, 0), returns)
        self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], UNKNOWN)
        m.mark_scan((0, 0), returns)
        self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], OCCUPIED)
        self.assertEqual(m.low_obstacles().shape, (1, 3))

    def test_adjacent_single_hits_do_not_replace_repeated_evidence(self):
        m = OccupancyGrid(resolution=.05, detect_low=True)
        m.mark_scan((0, 0), [[1.02, .02, .04], [1.07, .02, .04], [1.02, .07, .04]])
        self.assertNotEqual(m.grid[m.world_to_cell(1.02, .02)], OCCUPIED)

    def test_normal_floor_noise_does_not_become_an_obstacle(self):
        m = OccupancyGrid(resolution=.05, detect_low=True)
        noise = np.random.default_rng(17).normal(0, .01, 100)
        for batch in np.array_split(noise, 10):
            m.mark_scan((0, 0), [[1.02, .02, float(z)] for z in batch])
        self.assertEqual(m.grid[m.world_to_cell(1.02, .02)], FREE)
        self.assertEqual(len(m.low_obstacles()), 0)

    def test_one_high_floor_noise_sample_is_not_enough(self):
        m = OccupancyGrid(detect_low=True)
        m.mark_scan((0, 0), [[1., 0., .05]])
        self.assertNotEqual(m.grid[m.world_to_cell(1., 0.)], OCCUPIED)
        self.assertEqual(len(m.low_obstacles()), 0)

    def test_low_evidence_survives_growth_and_snapshot(self):
        m = OccupancyGrid(resolution=.05, size=100, grow=True, detect_low=True)
        m.mark_scan((0, 0), [[1.02, .02, .04]] * 8)
        before = m.low_obstacles().copy()
        m.ensure_bounds([[-12., -15.]])
        np.testing.assert_array_equal(m.snapshot().low_obstacles(), before)

    def test_uncertain_floor_evidence_never_clears_a_tall_obstacle(self):
        m = OccupancyGrid(detect_low=True)
        m.mark_scan((0, 0), [[1., 0., .04], [1., 0., .04], [1., 0., .3]])
        self.assertEqual(m.grid[m.world_to_cell(1., 0.)], OCCUPIED)


class RoomRegionTests(unittest.TestCase):
    def two_rooms(self):
        m = OccupancyGrid(resolution=.1, size=90, grow=True)
        m.grid[:] = OCCUPIED
        m.grid[10:80, 10:80] = FREE
        m.grid[10:80, 44:46] = OCCUPIED
        m.grid[40:50, 44:46] = FREE  # open, 1 m doorway
        return m

    def test_open_doorway_splits_two_room_cores(self):
        m = self.two_rooms()
        regions = RoomRegions().update(m)
        left, right = m.cell_to_world(25, 25), m.cell_to_world(25, 65)
        self.assertIsNotNone(regions.room_at(*left))
        self.assertNotEqual(regions.room_at(*left), regions.room_at(*right))
        self.assertIsNotNone(regions.room_at(*m.cell_to_world(45, 45)))

    def test_ids_and_explicit_names_survive_map_growth(self):
        m = self.two_rooms()
        regions = RoomRegions().update(m)
        point = m.cell_to_world(25, 25)
        regions.name_room(*point, "office")
        m.ensure_bounds([[-15., -12.]])
        regions.update(m)
        self.assertEqual(regions.room_at(*point), "office")

    def test_unobserved_and_occupied_positions_have_no_room(self):
        m = self.two_rooms()
        m.grid[20, 20] = UNKNOWN
        regions = RoomRegions().update(m)
        self.assertIsNone(regions.room_at(*m.cell_to_world(20, 20)))
        self.assertIsNone(regions.room_at(*m.cell_to_world(20, 44)))
        self.assertIsNone(regions.room_at(100, 100))

    def test_narrow_region_without_a_core_is_not_lost(self):
        m = OccupancyGrid(size=40)
        m.grid[:] = OCCUPIED
        m.grid[10:13, 10:30] = FREE
        regions = RoomRegions().update(m)
        self.assertIsNotNone(regions.room_at(*m.cell_to_world(11, 20)))

    def test_unnamed_geometry_does_not_invent_semantic_room_names(self):
        m = self.two_rooms()
        regions = RoomRegions().update(m)
        self.assertRegex(regions.room_at(*m.cell_to_world(25, 25)), r"^room_\d+$")


if __name__ == "__main__":
    unittest.main()
