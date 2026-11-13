"""Short real-physics regression; the five-minute acceptance is a separate evaluator."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAVE_SIM = (importlib.util.find_spec("mujoco") is not None and importlib.util.find_spec("torch") is not None
            and (ROOT/"_deps/unitree_mujoco/unitree_robots/go2/scene.xml").exists())


@unittest.skipUnless(HAVE_SIM, "MuJoCo and walking dependencies not installed")
class LocalisationSimulationTests(unittest.TestCase):
    def test_saved_place_return_with_real_lidar_and_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"result.json"
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
            process = subprocess.run([sys.executable, str(ROOT/"simulation/evaluate_drift.py"), "--seconds", "12",
                                      "--output", str(output)], capture_output=True, text=True, env=env, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout[-2500:]+process.stderr[-1500:])
            result = json.loads(output.read_text())
            self.assertTrue(result["passed"], result)
            self.assertLess(result["return_error_m"], .3)
            self.assertGreaterEqual(result["completed_corners"], 2)
