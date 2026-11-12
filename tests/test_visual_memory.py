"""Semantic object memory (go2/visual_memory.py) and its place in look, recall and approach, with a toy embedder: a model
in which a can looks a lot like a bottle, a cup a little, and a chair not at all. The real model has its own
tests (test_embedder)."""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2 import perception as P  # noqa: E402
from go2.adapters.base import CameraFrame  # noqa: E402
from go2.perception import Detection  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.tools import resolve  # noqa: E402
from go2.visual_memory import MIN_CROP_PX, VisualMemory, crop  # noqa: E402
from tests.fakes import FakeRobot, wait_for  # noqa: E402


def unit(*values):
    v = np.array(values + (0.,) * (8 - len(values)), np.float32)
    return v / np.linalg.norm(v)


CONCEPTS = {"bottle": unit(1), "can": unit(.92, .39), "cup": unit(.76, 0, .65), "chair": unit(0, 0, 0, 1),
            "couch": unit(0, 0, 0, .85, .53), "ball": unit(0, 0, 0, 0, 0, 1), "nothing": unit(0, 0, 0, 0, 0, 0, 0, 1)}
CODES = list(CONCEPTS)


class ToyEmbedder:
    """A crop's middle pixel says which concept it is a picture of; a phrase is the concept named by its last
    word (something else entirely if it names none)."""
    name, dim, threshold, strict_threshold = "toy-8", 8, .75, .9

    def __init__(self):
        self.image_batches, self.broken = [], False

    def images(self, crops):
        if self.broken:
            raise RuntimeError("the model fell over")
        self.image_batches.append(len(crops))
        return np.stack([CONCEPTS[CODES[int(c[c.shape[0] // 2, c.shape[1] // 2, 0])]] for c in crops])

    def texts(self, phrases):
        return np.stack([CONCEPTS.get(p.split()[-1].rstrip("s"), CONCEPTS["nothing"]) for p in phrases])

    def match(self, text_vectors, image_vectors):
        return np.clip(np.asarray(text_vectors) @ np.asarray(image_vectors).T, 0., 1.)


class StoreTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "room-visual.json"
        self.now = [1000.]
        self.memory = VisualMemory(self.path, ToyEmbedder(), clock=lambda: self.now[0])

    def add(self, concept, x, y, **kw):
        return self.memory.add(CONCEPTS[concept], x, y, **kw)

    def test_bottle_brings_back_the_bottle_then_the_can_then_the_cup_and_not_the_chair(self):
        self.add("chair", 0., 0., label="chair")
        self.add("cup", 1., 0., label="cup", color="white")
        self.add("can", 2., 0., label="cup", color="red")           # the detector had no better word for it
        self.add("bottle", 3., 0., label="bottle")
        found = self.memory.search("bottle")
        self.assertEqual([(e["x"], round(score, 2)) for e, score in found], [(3., 1.), (2., .92), (1., .76)])
        self.assertEqual(self.memory.search("a rubbish bin"), [])
        self.assertEqual([e["label"] for e, _ in self.memory.search("somewhere to sit on a chair")], ["chair"])

    def test_the_same_thing_seen_again_updates_its_entry(self):
        first = self.add("can", 2., 0., label="cup")
        self.now[0] += 60
        again = self.add("can", 2.2, .1, label="cup")
        self.assertEqual((len(self.memory), again["id"], again["seen"], again["t"]), (1, first["id"], 2, 1060.))
        self.assertEqual((again["x"], again["first"]), (2.2, 1000.))

    def test_something_different_at_the_same_spot_or_the_same_elsewhere_is_another_entry(self):
        self.add("can", 2., 0.)
        self.add("chair", 2.1, 0.)                 # a different thing where the can was
        self.add("can", 4., 0.)                    # another can somewhere else
        self.assertEqual(len(self.memory), 3)
        self.assertEqual(len(self.memory.search("can")), 2)

    def test_the_label_decides_what_is_the_same_thing_when_there_is_one(self):
        self.add("can", 2., 0., label="cup")
        self.add("bottle", 2.1, 0., label="bottle")        # looks much like it, but the detector says otherwise
        self.assertEqual(len(self.memory), 2)
        self.add("can", 2.1, 0., label="cup")              # the cup again, the same view of it
        self.assertEqual((len(self.memory), self.memory.entries[0]["seen"]), (2, 2))
        self.add("can", 2., .1)                            # no label: goes by look, to the thing most like it
        self.assertEqual((len(self.memory), self.memory.entries[0]["seen"]), (2, 3))

    def test_a_thing_keeps_a_few_different_views_of_itself_and_answers_once(self):
        views = ["can", "cup", "chair", "couch", "ball"]   # five pictures of one mug that look nothing alike
        for i, view in enumerate(views):
            self.now[0] += 1
            self.add(view, 2., 0., label="cup", pixels=100)
        self.assertEqual(len(self.memory), 4)              # the oldest view made way for the fifth
        self.assertEqual(sorted(e["label"] for e in self.memory.entries), ["cup"] * 4)
        self.assertEqual(len(self.memory.search("ball")), 1)
        self.assertEqual(len(self.memory.search("chair")), 1)      # two views fit (chair, couch): one place, one answer
        self.assertEqual(self.memory.search("can"), [])    # that view is the one that went

    def test_several_sightings_of_one_place_are_one_answer(self):
        self.add("bottle", 3., 0.)
        self.add("cup", 3.3, 0.)                   # beside it: within a step, so one place
        self.add("can", 6., 0.)
        self.assertEqual(len(self.memory), 3)
        self.assertEqual([e["x"] for e, _ in self.memory.search("bottle")], [3., 6.])

    def test_it_survives_a_restart_and_vectors_from_another_model_are_set_aside(self):
        self.add("can", 2., 0., label="cup", color="red", size_m=.1)
        self.memory.save(force=True)
        again = VisualMemory(self.path, ToyEmbedder())
        self.assertEqual((len(again), again.entries[0]["label"], again.search("bottle")[0][0]["id"]), (1, "cup", "v1"))
        self.assertEqual(again.add(CONCEPTS["chair"], 0., 0.)["id"], "v2")
        other = ToyEmbedder()
        other.name = "another-model"
        fresh = VisualMemory(self.path, other)
        self.assertEqual(len(fresh), 0)
        self.assertTrue(self.path.with_name(self.path.name + ".old").exists())

    def test_saving_is_spaced_out_and_nothing_is_written_without_a_path(self):
        self.add("can", 2., 0.)
        self.memory.save(force=True)
        self.add("chair", 0., 0.)
        self.memory.save()                          # too soon after the last write
        self.assertEqual(len(VisualMemory(self.path, ToyEmbedder())), 1)
        nowhere = VisualMemory(None, ToyEmbedder())
        nowhere.add(CONCEPTS["can"], 0., 0.)
        nowhere.save(force=True)
        self.assertEqual(len(nowhere), 1)

    def test_the_oldest_are_dropped_beyond_the_limit(self):
        with mock.patch("go2.visual_memory.MAX_ENTRIES", 3):
            for i in range(5):
                self.now[0] += 1
                self.add("can", float(i * 2), 0.)
        self.assertEqual([e["x"] for e in self.memory.entries], [4., 6., 8.])
        self.assertEqual(self.memory.vectors.shape, (3, 8))

    def test_a_phrase_is_only_embedded_once(self):
        calls = []
        self.memory.embedder.texts = lambda phrases: calls.append(phrases) or np.stack([CONCEPTS["can"]] * len(phrases))
        self.add("can", 0., 0.)
        self.memory.search("The  Can")
        self.memory.search("the can")
        self.assertEqual(calls, [["the can"]])


class CropTests(unittest.TestCase):
    def test_a_crop_takes_the_box_with_a_margin_and_stays_inside_the_picture(self):
        picture = np.zeros((100, 200, 3), np.uint8)
        picture[40:60, 100:140] = 7
        part = crop(picture, [.5, .4, .7, .6], margin=.25)
        self.assertEqual(part.shape, (30, 60, 3))
        self.assertEqual(int(part[5, 10, 0]), 7)
        self.assertEqual(crop(picture, [.9, .8, 1., 1.], margin=.5).shape, (30, 30, 3))

    def test_a_few_pixels_are_not_worth_a_vector(self):
        picture = np.zeros((100, 200, 3), np.uint8)
        self.assertIsNone(crop(picture, [.5, .5, .5 + (MIN_CROP_PX - 6) / 200, .9], margin=0.))


SLOTS = [[.05 + .18 * i, .3, .2 + .18 * i, .7] for i in range(5)]       # where each test thing is in the picture


class Looking(unittest.TestCase):
    """An open floor and a camera picture in which each thing's box is painted with the code of what it is a
    picture of (which may differ from what the detector calls it)."""

    def setUp(self):
        self.things = []                    # (label, colour, x, y): what the fake camera can see
        self.pictures = {}                  # label -> concept the thing really looks like (default: its label)
        self.events = []
        self.robot = FakeRobot(camera=True)
        self.robot.camera_period = .01
        self.robot.camera = self.camera
        self.robot.connect()
        self.addCleanup(self.robot.close)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.memory_path = Path(folder.name) / "room.json"
        self.executor = self.make_executor()
        self.embedder = ToyEmbedder()
        self.executor.use_visual_memory(self.embedder)
        patch = mock.patch.object(P, "fuse", lambda *a, **k: self.seen())
        patch.start()
        self.addCleanup(patch.stop)

    def make_executor(self):
        executor = Executor(on_event=self.events.append, memory_path=self.memory_path)
        self.addCleanup(executor.close)
        executor.attach(self.robot)
        return executor

    def camera(self):
        rgb = np.zeros((200, 400, 3), np.uint8)
        for i, (label, _, x, y) in enumerate(self.things):
            left, top, right, bottom = SLOTS[i]
            concept = self.pictures.get(label, label.split()[-1])
            code = CODES.index(concept) if concept in CODES else CODES.index("nothing")
            rgb[int(top * 200):int(bottom * 200), int(left * 400):int(right * 400)] = code
        return CameraFrame(rgb=rgb, masks=[], intrinsics={}, camera_pos=np.zeros(3), camera_rot=np.eye(3),
                           state=self.robot.state())

    def seen(self, hfov_deg=120., max_m=4.5):
        s, out = self.robot.state(), []
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        for i, (label, colour, x, y) in enumerate(self.things):
            dx, dy = c * (x - s.x) + sn * (y - s.y), -sn * (x - s.x) + c * (y - s.y)
            r, bearing = math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))
            if abs(bearing) > hfov_deg / 2 or not .25 < r < max_m:
                continue
            out.append(Detection(id=f"obj{i + 1}", label=label, color=colour, distance_m=r, bearing_deg=bearing,
                                 x_m=dx, y_m=dy, distance_source="lidar", confidence=.9, box=list(SLOTS[i])))
        return sorted(out, key=lambda d: d.distance_m)

    def call(self, name, args=None):
        return self.executor.submit(resolve(name, args or {}), "g" + name, 1)[0]

    def finished(self, count=1, timeout=90):
        self.assertTrue(wait_for(lambda: len(self.events) >= count, timeout), "the job did not finish")
        return self.events[count - 1]


class LookTests(Looking):
    def test_a_look_remembers_what_each_thing_looked_like_but_not_people(self):
        self.things = [("cup", "red", 1.5, .3), ("chair", "blue", 1.5, -.5), ("person", None, 2., 0.)]
        self.executor.look()
        memory = self.executor.visual
        self.assertEqual(sorted(e["label"] for e in memory.entries), ["chair", "cup"])
        cup = next(e for e in memory.entries if e["label"] == "cup")
        self.assertEqual((cup["color"], round(cup["x"], 1), round(cup["y"], 1)), ("red", 1.5, .3))
        self.executor.look()
        self.assertEqual((len(memory), cup["id"], memory.entries[0]["seen"]), (2, cup["id"], 2))
        self.assertEqual(self.embedder.image_batches, [2, 2])        # one batch a look

    def test_look_results_are_unchanged(self):
        self.things = [("cup", "red", 1.5, .3)]
        with_memory = self.executor.look()
        self.executor.visual = None
        without = self.executor.look()
        self.assertEqual(set(with_memory), set(without))
        self.assertEqual(with_memory["objects"], without["objects"])

    def test_a_model_that_fails_costs_the_memory_not_the_look(self):
        self.things = [("cup", "red", 1.5, .3)]
        self.embedder.broken = True
        result = self.executor.look()
        self.assertEqual(result["objects"][0]["label"], "cup")
        self.assertEqual(len(self.executor.visual), 0)

    def test_a_thing_that_moved_is_remembered_where_it_was_last_seen(self):
        self.things = [("cup", "red", 1.5, .6)]
        self.pictures = {"cup": "can"}
        self.executor.look()
        self.things = [("cup", "red", 1.5, -.9)]
        self.executor.look()
        thing, _, y = self.executor._recall("bottle")
        self.assertEqual((thing.label, round(y, 1)), ("cup", -.9))


class RecallTests(Looking):
    def test_a_word_the_label_lacks_recalls_what_looked_like_it(self):
        self.things = [("cup", "red", 1.5, .3), ("chair", "blue", 1.5, -.5)]
        self.pictures = {"cup": "can"}
        self.executor.look()
        thing, x, y = self.executor._recall("the bottle")
        self.assertEqual((thing.label, round(x, 1), round(y, 1)), ("cup", 1.5, .3))

    def test_a_label_that_fits_comes_first(self):
        self.things = [("cup", "red", 1.5, .3), ("bottle", "green", 1.5, -.5)]
        self.pictures = {"cup": "bottle", "bottle": "chair"}      # the cup looks more like a bottle than the bottle
        self.executor.look()
        thing, _, y = self.executor._recall("bottle")
        self.assertEqual((thing.label, round(y, 1)), ("bottle", -.5))

    def test_something_never_seen_is_not_recalled(self):
        self.things = [("chair", "blue", 1.5, -.5)]
        self.executor.look()
        self.assertIsNone(self.executor._recall("bottle"))

    def test_it_is_remembered_after_a_restart(self):
        self.things = [("cup", "red", 1.5, .3)]
        self.pictures = {"cup": "can"}
        self.executor.look()
        self.executor.close()
        restarted = self.make_executor()
        restarted.use_visual_memory(ToyEmbedder())
        thing, x, y = restarted._recall("bottle")
        self.assertEqual((thing.label, round(x, 1), round(y, 1)), ("cup", 1.5, .3))

    def test_without_a_visual_memory_nothing_changes(self):
        self.executor.visual = None
        self.things = [("cup", "red", 1.5, .3)]
        self.pictures = {"cup": "can"}
        self.executor.look()
        self.assertIsNone(self.executor._recall("bottle"))


class ApproachTests(Looking):
    def test_approach_the_can_takes_the_thing_that_looks_like_one(self):
        self.things = [("bottle", "green", 1.4, .4), ("chair", "blue", 1.4, -.6)]
        self.pictures = {"bottle": "can"}
        self.call("approach", {"target": "can"})
        task = self.finished()
        self.assertEqual((task.status, task.detail["target"]["label"]), ("completed", "bottle"), task.detail)
        self.assertGreater(self.robot.y, 0.)                    # it walked to the bottle, not the chair

    def test_approach_turns_toward_a_remembered_look_alike_out_of_view(self):
        self.things = [("bottle", "green", 1.4, .4)]
        self.pictures = {"bottle": "can"}
        self.executor.look()
        self.robot.yaw = math.pi                                # facing away from it now
        self.call("approach", {"target": "can"})
        task = self.finished()
        self.assertEqual((task.status, task.detail["target"]["label"]), ("completed", "bottle"), task.detail)
        self.assertLess(math.hypot(self.robot.x - 1.4, self.robot.y - .4), 1.)

    def test_a_remembered_look_alike_is_followed_by_its_label_once_recalled(self):
        self.things = [("bottle", "green", 1.4, .4)]
        self.pictures = {"bottle": "can"}
        self.executor.look()
        self.robot.yaw = math.pi
        self.pictures = {"bottle": "chair"}                     # from now on its picture doesn't fit the words
        self.call("approach", {"target": "can"})
        task = self.finished()
        self.assertEqual((task.status, task.detail["target"]["label"]), ("completed", "bottle"), task.detail)

    def test_nothing_that_looks_like_it_is_still_not_found(self):
        self.things = [("chair", "blue", 1.4, -.6)]
        self.call("approach", {"target": "can"})
        task = self.finished()
        self.assertEqual(task.status, "failed")
        self.assertIn("can't find a", task.detail["error"])


class TranslationTests(Looking):
    """Words that fit no label are taken to mean the closest detector label, and Go2 says which."""

    def test_a_word_with_no_label_is_taken_to_mean_the_closest_one_and_it_says_so(self):
        self.things = [("bottle", "green", 1.4, .4)]
        result = self.call("approach", {"target": "the green can"})
        self.assertEqual(result["looking_for"], "green bottle")
        self.assertIn("approach the green bottle (what I take 'the green can' to mean)", result["action"])
        task = self.finished()
        self.assertEqual((task.status, task.detail["target"]["label"]), ("completed", "bottle"), task.detail)

    def test_words_that_already_fit_a_label_are_left_alone(self):
        result = self.call("approach", {"target": "the cup"})
        self.assertNotIn("looking_for", result)
        self.assertEqual(self.executor._label_for("sofa"), None)        # its synonym fits: couch

    def test_no_clear_label_means_no_translation(self):
        self.assertIsNone(self.executor._label_for("guitar"))          # every label is as far from it
        self.executor.visual = None
        self.assertIsNone(self.executor._label_for("can"))             # and nothing without the memory

    def test_a_translation_not_found_says_what_it_was_taken_for(self):
        self.things = [("chair", "blue", 1.4, -.6)]
        self.call("approach", {"target": "can"})
        task = self.finished()
        self.assertEqual(task.status, "failed")
        self.assertIn("can't find a bottle (what I took 'can' to mean)", task.detail["error"])


class SpeedTests(unittest.TestCase):
    def test_searching_ten_thousand_things_is_quick(self):
        import time
        rng = np.random.default_rng(0)
        memory = VisualMemory(None, ToyEmbedder())
        vectors = rng.normal(size=(10000, 8)).astype(np.float32)
        memory.entries = [{"id": f"v{i}", "t": 0., "x": float(i % 100), "y": float(i // 100), "label": ""}
                          for i in range(10000)]
        memory.vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float16)
        memory.search("bottle")                                 # the phrase's vector, once
        began = time.perf_counter()
        memory.search("bottle")
        self.assertLess(time.perf_counter() - began, .05)


if __name__ == "__main__":
    unittest.main()
