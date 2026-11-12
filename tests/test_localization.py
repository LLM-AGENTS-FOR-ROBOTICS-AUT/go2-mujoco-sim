import math
import unittest
import numpy as np

from go2.localization import ScanLocalizer, inverse, transform, wrap
from simulation.odometry import DriftConfig, DriftingOdometry


def room_scan(pose):
    x, y = np.linspace(-4., 4., 130), np.linspace(-3., 3., 100)
    walls = np.vstack([np.column_stack([x, np.full(len(x), z)]) for z in (-3., 3.)] +
                      [np.column_stack([np.full(len(y), z), y]) for z in (-4., 4.)])
    local = transform(walls, inverse(pose))
    return np.column_stack([local, np.full(len(local), .4)])


class LocalisationTests(unittest.TestCase):
    def test_drift_is_repeatable_and_zero_configuration_is_identity(self):
        for config in (DriftConfig(seed=12), DriftConfig(0., 0., 0., 0., 0)):
            a, b = DriftingOdometry(config), DriftingOdometry(config)
            for t in np.linspace(0., 5., 101):
                truth = np.array([math.sin(t), 1-math.cos(t), wrap(t)])
                np.testing.assert_array_equal(a.update(truth), b.update(truth))
            if config.translation_scale == 0.:
                np.testing.assert_allclose(a.pose, truth, atol=1e-12)
            else:
                self.assertGreater(np.linalg.norm(a.pose[:2]-truth[:2]), .01)

    def test_sensor_matching_corrects_accumulated_drift_on_revisits(self):
        odometry = DriftingOdometry(DriftConfig(seed=3))
        localizer = ScanLocalizer()
        errors = []
        for i, theta in enumerate(np.linspace(0., 4*math.pi, 100)):
            truth = np.array([math.sin(theta), 1-math.cos(theta), wrap(theta)])
            raw = odometry.update(truth)
            corrected = localizer.update(raw, room_scan(truth), float(i))
            errors.append(float(np.linalg.norm(corrected[:2]-truth[:2])))
        self.assertEqual(localizer.status()["state"], "tracking")
        self.assertLess(max(errors), .1)
        self.assertGreater(np.linalg.norm(raw[:2]-truth[:2]), .1)

    def test_rejected_scans_do_not_create_second_walls(self):
        localizer = ScanLocalizer()
        localizer.update([0., 0., 0.], room_scan([0., 0., 0.]), 0.)
        size = localizer.status()["reference_points"]
        for i in range(3):
            localizer.update([5., 5., 1.], room_scan([0., 0., 0.]), i+1.)
        self.assertEqual(localizer.status()["state"], "lost")
        self.assertEqual(localizer.status()["reference_points"], size)
        localizer.update([0., 0., 0.], room_scan([0., 0., 0.]), 4.)
        self.assertEqual(localizer.status()["state"], "tracking")

    def test_sparse_and_out_of_order_scans_cannot_change_transform(self):
        localizer = ScanLocalizer()
        localizer.update([0., 0., 0.], room_scan([0., 0., 0.]), 2.)
        before = localizer.map_from_odom.copy()
        localizer.update([1., 2., 1.], room_scan([0., 0., 0.]), 1.)
        np.testing.assert_array_equal(localizer.map_from_odom, before)
        localizer.update([0., 0., 0.], np.zeros((0, 3)), 3.)
        self.assertFalse(localizer.status()["accepted"])

    def test_configuration_rejects_nonfinite_noise_and_unbounded_bias(self):
        for args in (dict(translation_scale=float("nan")), dict(yaw_bias_per_m=2.),
                     dict(translation_noise=-.01), dict(seed=True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                DriftConfig(**args)
