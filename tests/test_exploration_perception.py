import math
import unittest

import numpy as np
from go2 import perception as P


class MovingEnvelopeTests(unittest.TestCase):
    footprint=(.42,.23)

    def test_leg_obstacle_outside_standing_corridor_is_inside_moving_corridor(self):
        points=np.array([[.8,.225,.2]])
        self.assertEqual(P.obstacle_along(points,1.,0.),math.inf)
        self.assertAlmostEqual(P.obstacle_along(points,1.,0.,footprint=self.footprint),.38)

    def test_drop_and_observation_checks_use_the_same_wider_envelope(self):
        drop=np.array([[.8,.225,-.3]])
        self.assertEqual(P.drop_along(drop,1.,0.),math.inf)
        self.assertEqual(P.drop_along(drop,1.,0.,footprint=self.footprint),0.)
        floor=np.array([[x,.225,0.] for x in np.arange(.5,1.,.1)])
        self.assertEqual(P.observed_along(floor,1.,0.),0.)
        self.assertGreater(P.observed_along(floor,1.,0.,footprint=self.footprint),.3)

    def test_turning_circle_reserves_the_moving_legs(self):
        points=np.array([[.46,0.,.2]])
        self.assertGreater(P.turn_clearance(points),.03)
        self.assertEqual(P.turn_clearance(points,footprint=self.footprint),0.)

    def test_diagonal_motion_ignores_only_corners_outside_the_swept_body(self):
        u=math.sqrt(.5)
        away=np.array([[.4,-.3,.2]])
        toward=np.array([[.4,.3,.2]])
        self.assertEqual(P.obstacle_along(away,u,u,footprint=self.footprint),math.inf)
        self.assertAlmostEqual(P.obstacle_along(toward,u,u,footprint=self.footprint),.07/u)

    def test_reverse_sweep_keeps_points_behind_the_travel_direction_out(self):
        points=np.array([[-1.,0.,.2]])
        self.assertEqual(P.obstacle_along(points,1.,0.,footprint=self.footprint),math.inf)
        self.assertAlmostEqual(P.obstacle_along(points,-1.,0.,footprint=self.footprint),.58)

    def test_invalid_envelope_cannot_disable_clearance(self):
        for footprint in ((0.,.2),(-.4,.2),(math.nan,.2),(.4,)):
            with self.assertRaises(ValueError):
                P.obstacle_along(np.array([[1.,0.,.2]]),1.,0.,footprint=footprint)


if __name__=='__main__':
    unittest.main()
