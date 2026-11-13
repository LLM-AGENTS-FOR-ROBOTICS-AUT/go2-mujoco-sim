import unittest
from types import SimpleNamespace

from simulation.evaluate_exploration import FloorTruth, grid_scan, settle_robot, remember_attempt
from go2.room_map import FREE, OccupancyGrid


class CoverageMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.world = {"scene": "house", "objects": [], "doors": [], "start": [-1., 0., 0.],
                      "rooms": [{"name": "left", "type": "bedroom", "rect": [-2., -2., 0., 2.]},
                                {"name": "right", "type": "kitchen", "rect": [0., -2., 2., 2.]}]}
        self.truth = FloorTruth(self.world)

    def test_unseen_room_counts_in_denominator(self):
        observed = OccupancyGrid(resolution=.05, size=160)
        for _ in range(8):
            grid_scan(self.truth, observed, self.world["start"])
        self.assertGreater(self.truth.coverage(observed), .4)
        self.assertLess(self.truth.coverage(observed), .6)
        self.assertNotEqual(observed.grid[observed.world_to_cell(1., 0.)], FREE)

    def test_empty_observations_have_zero_coverage(self):
        self.assertEqual(self.truth.coverage(OccupancyGrid()), 0.)

    def test_scoring_does_not_change_observations(self):
        observed = OccupancyGrid()
        before = observed.grid.copy()
        self.truth.coverage(observed)
        self.assertTrue((observed.grid == before).all())

    def test_contact_during_braking_is_included_in_final_state(self):
        class Clock:
            t = 0.
            def monotonic(self):
                return self.t
            def sleep(self, seconds):
                self.t += seconds
        clock = Clock()
        sim = SimpleNamespace(stop_move=lambda: None,
                              state=lambda: SimpleNamespace(speed=.2 if clock.t < .1 else 0., wz=0.,
                                                            posture="standing", extra={"collision_count": int(clock.t >= .1)}))
        state, settled, duration = settle_robot(sim, clock)
        self.assertTrue(settled)
        self.assertGreaterEqual(duration, .1)
        self.assertEqual(state.extra["collision_count"], 1)


class AttemptMemoryTests(unittest.TestCase):
    def test_timeout_after_travel_does_not_mark_unreached_room_visited(self):
        visited,blocked=[],[(8.,0.,0.)]
        before=SimpleNamespace(x=0.,y=0.,yaw=0.)
        after=SimpleNamespace(x=2.,y=0.,yaw=.4)
        remember_attempt(visited,blocked,(5.,0.,0.),before,after,'failed')
        self.assertEqual(visited,[(2.,0.,.4)])
        self.assertNotIn((5.,0.,0.),visited)
        self.assertEqual(blocked,[])

    def test_blocked_attempts_have_a_bounded_temporary_exclusion(self):
        visited,blocked=[],[]
        state=SimpleNamespace(x=0.,y=0.,yaw=0.)
        for i in range(5):
            remember_attempt(visited,blocked,(i+1.,0.,0.),state,state,'blocked')
        self.assertEqual(visited,[])
        self.assertEqual(blocked,[(3.,0.,0.),(4.,0.,0.),(5.,0.,0.)])

    def test_completed_rotation_records_its_actual_observation_pose(self):
        visited,blocked=[],[(8.,0.,0.)]
        before=SimpleNamespace(x=0.,y=0.,yaw=0.)
        after=SimpleNamespace(x=0.,y=0.,yaw=1.)
        remember_attempt(visited,blocked,(.1,.1,1.),before,after,'completed')
        self.assertEqual(visited,[(0.,0.,1.)])
        self.assertEqual(blocked,[])


if __name__ == "__main__":
    unittest.main()
