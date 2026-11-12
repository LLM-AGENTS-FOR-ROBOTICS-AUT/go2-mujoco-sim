"""Image-text embeddings for the visual memory (go2.embedder).

The pure parts (errors, the decision rule's arithmetic, preprocessing, the download's checksum handling) run
everywhere. The model tests need torch, transformers and the pinned SigLIP 2 weights in models/ (python -m
go2.embedder fetch); the photo test also needs ultralytics' sample images and the simulator test MuJoCo rendering.
Each is skipped, not failed, where those are missing."""
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2 import embedder as E  # noqa: E402
from tests.fakes import needs_camera  # noqa: E402

READY = E.Embedder().available()          # None when the model can be loaded here
SAMPLES = None
try:
    import ultralytics
    SAMPLES = Path(ultralytics.__file__).parent / "assets"
except ImportError:
    pass


def unit(rows):
    rows = np.asarray(rows, np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class WithoutTheModelTests(unittest.TestCase):
    def test_importing_needs_only_numpy(self):
        code = ("import sys; sys.modules.update(torch=None, transformers=None, PIL=None); "
                "sys.path.insert(0, sys.argv[1]); from go2 import embedder; e = embedder.Embedder(); "
                "print(e.name, e.dim, e.threshold); print(e.available())")
        done = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        name, dim, threshold, reason = done.stdout.split(None, 3)
        self.assertEqual((name, dim, threshold), ("siglip2-b16", "768", "0.5"))
        self.assertIn("pip install", reason)

    def test_dimension_and_threshold_are_known_without_loading(self):
        for name, dim in (("siglip2-b16", 768), ("siglip2-b16-192", 768), ("siglip-b16", 768)):
            e = E.Embedder(name)
            self.assertEqual((e.name, e.dim, e.threshold), (name, dim, .5))
            self.assertTrue(.5 < e.strict_threshold < 1)
            self.assertIsNone(e._model)

    def test_unknown_model_is_an_embedder_error(self):
        with self.assertRaises(E.EmbedderError) as caught:
            E.Embedder("clip-giant")
        self.assertIn("siglip2-b16", str(caught.exception))

    def test_missing_weights_say_how_to_get_them(self):
        with tempfile.TemporaryDirectory() as folder:
            e = E.Embedder()
            e.folder = Path(folder)
            reason = e.available()
            self.assertIsInstance(reason, str)
            if not e._missing_packages():
                self.assertIn("python -m go2.embedder fetch", reason)
                self.assertIn("model.safetensors", reason)
            crop = np.zeros((40, 30, 3), np.uint8)
            for call in (lambda: e.images([crop]), lambda: e.texts(["bottle"]), e.load):
                with self.assertRaises(E.EmbedderError) as caught:
                    call()
                self.assertEqual(str(caught.exception), reason)
            self.assertEqual(list(Path(folder).iterdir()), [], "available() and a failed load download nothing")

    def test_a_wrong_file_is_refused_at_load(self):
        if E.Embedder()._missing_packages():
            self.skipTest("torch / transformers not installed")
        with tempfile.TemporaryDirectory() as folder:
            e = E.Embedder()
            e.folder = Path(folder)
            for name, (size, _) in e.spec["files"].items():          # right sizes, wrong contents (sparse)
                with open(Path(folder) / name, "wb") as f:
                    f.truncate(size)
            self.assertIsNone(e.available())                        # cheap check: sizes only
            with self.assertRaises(E.EmbedderError) as caught:
                e.load()
            self.assertIn("differs from the pinned file", str(caught.exception))

    def test_empty_inputs(self):
        e = E.Embedder()
        self.assertEqual(e.images([]).shape, (0, 768))
        self.assertEqual(e.texts([]).shape, (0, 768))
        self.assertEqual(e.match(np.zeros((0, 768)), unit(np.ones((3, 768)))).shape, (0, 3))

    def test_match_is_the_calibrated_cosine(self):
        e = E.Embedder()
        rng = np.random.default_rng(0)
        texts, images = unit(rng.normal(size=(4, 768))), unit(rng.normal(size=(6, 768)))
        scores = e.match(texts, images)
        self.assertEqual((scores.shape, scores.dtype), ((4, 6), np.float32))
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        cos = texts @ images.T
        order = np.argsort(cos.ravel())
        self.assertTrue((np.diff(scores.ravel()[order]) >= 0).all(), "monotonic in the cosine")
        # 0.5 exactly at the calibrated boundary; float16 storage gives the same decisions
        boundary = e.spec["boundary"]
        image = unit([[1.] + [0.] * 767])
        text = np.array([[boundary, np.sqrt(1 - boundary ** 2)] + [0.] * 766], np.float32)
        self.assertAlmostEqual(float(e.match(text, image)[0, 0]), .5, places=3)
        self.assertTrue(((e.match(texts.astype(np.float16), images.astype(np.float16)) >= e.threshold)
                         == (scores >= e.threshold)).all())
        self.assertEqual(e.match(texts[0], images[0]).shape, (1, 1))

    def test_prompts(self):
        self.assertEqual(E.normalise_phrase("  Teddy   BEAR "), "teddy bear")
        self.assertEqual(E.prompt("bottle", "{}"), "bottle")
        siglip = E.MODELS["siglip-b16"]["template"]
        self.assertEqual(E.prompt("apple", siglip), "this is a photo of an apple.")
        self.assertEqual(E.prompt("my shoes", siglip), "this is a photo of my shoes.")
        self.assertEqual(E.prompt("something to drink from", siglip), "this is a photo of something to drink from.")
        self.assertEqual(E.prompt("food", siglip), "this is a photo of food.")

    def test_preprocessing_pads_to_a_square_and_is_deterministic(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("pillow not installed")
        crop = np.zeros((20, 60, 3), np.uint8)
        crop[..., 0] = 255
        out = E.square(crop, 30)
        self.assertEqual((out.shape, out.dtype), ((30, 30, 3), np.float32))
        np.testing.assert_allclose(out[0, 15], [128 / 255] * 3, atol=1e-6)        # grey above and below
        np.testing.assert_allclose(out[15, 15], [1, 0, 0], atol=1e-6)            # the red crop in the middle
        np.testing.assert_array_equal(out, E.square(crop.copy(), 30))
        self.assertEqual(E.square(np.zeros((5, 5), np.uint8), 8).shape, (8, 8, 3))   # grey -> RGB
        self.assertEqual(E.square(np.zeros((5, 5, 4), np.uint8), 8).shape, (8, 8, 3))  # alpha dropped
        with self.assertRaises(ValueError):
            E.square(np.zeros((5, 5, 3), np.float32), 8)
        with self.assertRaises(ValueError):
            E.square(np.zeros((0, 5, 3), np.uint8), 8)


class FetchTests(unittest.TestCase):
    """fetch() against a fake server: checksums, no half-written files, nothing fetched twice."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.files = {"config.json": b'{"a": 1}', "model.safetensors": b"weights" * 1000}
        spec = dict(E.MODELS["siglip-b16"], folder="tiny",
                    files={k: (len(v), hashlib.sha256(v).hexdigest()) for k, v in self.files.items()})
        patcher = mock.patch.dict(E.MODELS, {"tiny": spec})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.requests = []

    def serve(self, corrupt=()):
        def urlopen(url, timeout=None):
            name = url.rsplit("/", 1)[1]
            self.requests.append(url)
            body = self.files[name]
            return io.BytesIO(b"x" + body[1:] if name in corrupt else body)
        return mock.patch("urllib.request.urlopen", urlopen)

    def test_downloads_checks_and_skips_what_is_there(self):
        e = E.Embedder("tiny")
        e.folder = Path(self.folder.name)
        seen = []
        with self.serve():
            self.assertEqual(e.fetch(lambda done, total, name: seen.append((done, total, name))), e.folder)
        self.assertEqual({p.name for p in e.folder.iterdir()}, set(self.files) | {".verified.json"})
        self.assertEqual(seen[-1][:2], (sum(map(len, self.files.values())),) * 2)
        self.assertTrue(all("/resolve/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/" in u for u in self.requests))
        before = len(self.requests)
        with self.serve():
            e.fetch()
        self.assertEqual(len(self.requests), before, "verified files are not downloaded again")

    def test_a_corrupt_download_is_refused_and_removed(self):
        e = E.Embedder("tiny")
        e.folder = Path(self.folder.name)
        with self.serve(corrupt={"model.safetensors"}), self.assertRaises(E.EmbedderError) as caught:
            e.fetch()
        self.assertIn("checksum", str(caught.exception))
        self.assertEqual({p.name for p in e.folder.iterdir()}, {"config.json", ".verified.json"})

    def test_network_failure_is_an_embedder_error(self):
        e = E.Embedder("tiny")
        e.folder = Path(self.folder.name)
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no route to host")), \
                self.assertRaises(E.EmbedderError) as caught:
            e.fetch()
        self.assertIn("internet connection", str(caught.exception))
        self.assertEqual([p.name for p in e.folder.iterdir()], [])


@unittest.skipIf(READY, f"visual memory model not available: {READY}")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = E.Embedder().load()
        rng = np.random.default_rng(1)
        # varied sizes and contents, as detection boxes come
        cls.crops = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((30, 45), (120, 80), (400, 300))]
        cls.crops.append(np.full((64, 64, 3), (200, 30, 30), np.uint8))

    def test_vectors_are_unit_float32(self):
        v = self.e.images(self.crops)
        t = self.e.texts(["bottle", "something to drink from"])
        for x, n in ((v, 4), (t, 2)):
            self.assertEqual((x.shape, x.dtype), ((n, 768), np.float32))
            np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)

    def test_deterministic(self):
        np.testing.assert_array_equal(self.e.images(self.crops), self.e.images([c.copy() for c in self.crops]))
        a = self.e.texts(["a red ball"])
        self.e._texts.clear()                     # computed again, not from the cache
        np.testing.assert_array_equal(a, self.e.texts(["a red ball"]))
        np.testing.assert_array_equal(a, self.e.texts(["  A Red   BALL"]))

    def test_batch_equals_one_by_one(self):
        together = self.e.images(self.crops * 3)              # 12 crops: more than one GPU batch
        alone = np.concatenate([self.e.images([c]) for c in self.crops])
        self.assertGreater(float(np.min(np.sum(together[:4] * alone, 1))), .999)
        self.assertGreater(float(np.min(np.sum(together[8:] * alone, 1))), .999)

    @unittest.skipUnless(SAMPLES and (SAMPLES / "bus.jpg").exists(), "ultralytics' sample photos not installed")
    def test_photos_retrieve_and_the_rule_says_no_to_what_is_not_there(self):
        from PIL import Image
        bus = np.asarray(Image.open(SAMPLES / "bus.jpg").convert("RGB"))
        zidane = np.asarray(Image.open(SAMPLES / "zidane.jpg").convert("RGB"))
        # boxes from YOLO11n-seg on these photos: the bus, the man on the right, his tie
        crops = [bus[229:726, 14:802], zidane[55:709, 703:1154], zidane[436:716, 442:518]]
        scores = self.e.match(self.e.texts(["bus", "a man in a suit", "tie"]), self.e.images(crops))
        self.assertEqual(list(scores.argmax(1)), [0, 1, 2], scores)
        self.assertGreaterEqual(scores[0, 0], self.e.threshold, scores)
        self.assertGreaterEqual(scores[2, 2], self.e.threshold, scores)
        absent = self.e.match(self.e.texts(["a refrigerator", "teddy bear", "a bowl of soup"]), self.e.images(crops))
        self.assertTrue((absent < self.e.threshold).all(), absent)
        # Measured, and a limit to know: the threshold was fitted on household objects, and generic words for
        # people or vehicles score low ("person" 0.36 on the man, "a vehicle" 0.03 on the bus) though they rank
        # the right crop first. docs/visual-memory-models.md, "Limits".

    @needs_camera
    def test_simulator_crops(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        import base64
        from go2.adapters.sim import SimAdapter
        sim = SimAdapter(python=sys.executable, scene="room", headless=True)
        sim.connect()
        crops = {}
        try:
            # Two fixed views in the room: the potted plant / couch side, and the teddy bear / chair side
            for x, y, yaw in ((-1.5, 1.2, 1.83), (1.0, -0.9, -1.35)):
                sim.reset(x, y, yaw)
                raw = sim._rpc("camera", timeout=8.0, full=True)
                h, w = raw["height"], raw["width"]
                rgb = np.frombuffer(base64.b64decode(raw["rgb"]), np.uint8).reshape(h, w, 3)
                index = np.frombuffer(base64.b64decode(raw["index"]), np.uint8).reshape(h, w)
                for obj in raw["objects"]:
                    ys, xs = np.nonzero(index == obj["index"])
                    if obj["label"] in ("potted plant", "couch", "teddy bear", "chair") and len(xs) > 400 \
                            and obj["label"] not in crops:
                        crops[obj["label"]] = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        finally:
            sim.close()
        if len(crops) < 3:
            self.skipTest(f"only {sorted(crops)} in view; the room's layout has changed")
        labels = sorted(crops)
        scores = self.e.match(self.e.texts(labels), self.e.images([crops[l] for l in labels]))
        self.assertEqual(list(scores.argmax(0)), list(range(len(labels))), (labels, scores))
        self.assertTrue((self.e.match(self.e.texts(["a refrigerator", "a toilet"]),
                                      self.e.images([crops[l] for l in labels])) < self.e.threshold).all())


if __name__ == "__main__":
    unittest.main()
