import time

from go2.object_memory import ObjectMemory


def test_memory_survives_reload(tmp_path):
    save_path = tmp_path / "memory.json"

    memory = ObjectMemory(save_path)
    memory.remember("chair", 1.5, 2.0, color="blue")

    reloaded = ObjectMemory(save_path)
    result = reloaded.recall("chair")

    assert len(result) == 1
    assert result[0]["x"] == 1.5
    assert result[0]["y"] == 2.0
    assert result[0]["color"] == "blue"


def test_memory_matches_labels_case_insensitively(tmp_path):
    memory = ObjectMemory(tmp_path / "memory.json")
    memory.remember("Chair", 1.0, 2.0)

    assert len(memory.recall("chair")) == 1


def test_memory_updates_moved_object_position(tmp_path):
    memory = ObjectMemory(tmp_path / "memory.json")

    memory.remember("chair", 1.0, 2.0, color="blue")
    memory.remember("chair", 4.0, 5.0, color="blue")

    result = memory.recall("chair")

    assert len(result) == 1
    assert result[0]["x"] == 4.0
    assert result[0]["y"] == 5.0


def test_memory_updates_seen_count_and_timestamp(tmp_path):
    memory = ObjectMemory(tmp_path / "memory.json")

    memory.remember("chair", 1.0, 2.0, color="blue")
    first = memory.recall("chair")[0]
    first_timestamp = first["last_seen"]

    time.sleep(0.01)

    memory.remember("chair", 1.0, 2.0, color="blue")
    updated = memory.recall("chair")[0]

    assert updated["times_seen"] == 2
    assert updated["last_seen"] > first_timestamp


