"""Download the realistic furniture used in the houses (house.py) and convert it for MuJoCo (~50 MB).

Sources: Poly Haven (polyhaven.com), CC0: free for any use, no attribution required; and a few CC-BY 4.0
Sketchfab models through Objaverse (allenai/objaverse on Hugging Face), which require their authors'
credit (CREDITS; THIRD_PARTY_NOTICES.md repeats it). Every file is pinned by URL and SHA-256, like the
repository's other downloads; a file that doesn't match is refused. Each glTF or GLB model becomes one OBJ mesh per texture or colour, turned Z-up,
centred on x, y = 0 and standing on z = 0, plus model.json with its size and credit. Without these files
the houses use plain shapes instead. The setup scripts run this with --agent / -Agent:

  .venv/bin/python simulation/fetch_furniture.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

DEST = Path(__file__).resolve().parents[1] / "_deps/furniture"
OBJAVERSE = "https://huggingface.co/datasets/allenai/objaverse/resolve/556637099bf4fa79ea7b239d0d9c328b8a2e9ac8"
# model -> [(file, url, sha256)]: the 1k glTF, its geometry and its colour textures (as PNG, which MuJoCo loads)
MODELS = {
    "dining_chair_02": [
        ('dining_chair_02.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/dining_chair_02/dining_chair_02_1k.gltf', '88352a44a17319846f61fe03ce75c79fb1849aa508b6a262366c149c1bbd06b8'),
        ('dining_chair_02.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/dining_chair_02/dining_chair_02.bin', 'c7f1ef858f73815176d2ab304472a69ccf6097e971e5258eb924a75f841d4af5'),
        ('textures/dining_chair_02_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/dining_chair_02/dining_chair_02_diff_1k.png', '27adbae3ca17f1247f1840f24e3d519514894d7ba67883eb806b553569494c21'),
    ],
    "painted_wooden_chair_01": [
        ('painted_wooden_chair_01.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/painted_wooden_chair_01/painted_wooden_chair_01_1k.gltf', '34f55ebe1851b89c045dc62d4461c848de3b93c0668e752a2b5f6cbed2d91a5e'),
        ('painted_wooden_chair_01.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/painted_wooden_chair_01/painted_wooden_chair_01.bin', 'eaf952048db1e770c45898049b8abe28deb83d3dbbaad81d5e7e43e22a322295'),
        ('textures/painted_wooden_chair_01_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/painted_wooden_chair_01/painted_wooden_chair_01_diff_1k.png', '47490b6013e5874121c995a9829449b3f9b730220b8f57eb023fee716d9c5b3d'),
    ],
    "potted_plant_02": [
        ('potted_plant_02.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/potted_plant_02/potted_plant_02_1k.gltf', 'c533dcce0d1aa4379eb7db04601a04983f8df2422b28578de922655a416e64a9'),
        ('potted_plant_02.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/potted_plant_02/potted_plant_02.bin', '9c9812482c78a59789ed32fa842b4cbd9365e09c900c31ebb998e5efe627fe97'),
        ('textures/potted_plant_02_leaves_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_02/potted_plant_02_leaves_diff_1k.png', 'c664ad56c2117d82ea3b6907d0e6554f1d627da05e0f3e28f71e31bea98c667f'),
        ('textures/potted_plant_02_pot_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_02/potted_plant_02_pot_diff_1k.png', '8988e4ce4d985162e5753b9864eb0971c72532d6f1e9914a513b04570a5b3449'),
    ],
    "potted_plant_01": [
        ('potted_plant_01.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/potted_plant_01/potted_plant_01_1k.gltf', '23b30294407684b02f14800d6685bd8938da5d2111e17753ac8dd3c37d8c88a0'),
        ('potted_plant_01.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/potted_plant_01/potted_plant_01.bin', 'a8dd1f7ce75f50dc25bbedc20809da02ee4193815c155b883a329925fa5548f7'),
        ('textures/potted_plant_01_leaves_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_01/potted_plant_01_leaves_diff_1k.png', '83d339ea0c13609fff151afd778d5625bc42eacf5e60454eedabf0b571b0bbaf'),
        ('textures/potted_plant_01_pot_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_01/potted_plant_01_pot_diff_1k.png', 'db986d79c9af9d10463ba3c8400bfaa7d2271e0a94cd0607b07e29e7053be3a8'),
    ],
    "Television_01": [
        ('Television_01.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/Television_01/Television_01_1k.gltf', 'ef4376366ea956519c1a1d06146a8e8cc2bb0fcbd5c6b9f919a54ce31d05d530'),
        ('Television_01.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/Television_01/Television_01.bin', 'fc32dd3e6722cfa093910d8b5b4e6173b0c92a13d58b55b8e9c6007a9218b0a0'),
        ('textures/Television_01_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/Television_01/Television_01_diff_1k.png', '4c4f5ce5bf0e5c67a6c7a373274861fb3c5da73b7e096635e5ab267be3ea3528'),
    ],
    "electric_stove": [
        ('electric_stove.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/electric_stove/electric_stove_1k.gltf', '3be4d1b9ec0ead3c41ea8ef09d61e3b7611b69cbeea26a40ea87355acb0b3c55'),
        ('electric_stove.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/electric_stove/electric_stove.bin', '66f210539346b8f339b7519664ad6fa9b74d0932d8445b6b7e40eabcb40cd4f9'),
        ('textures/electric_stove_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/electric_stove/electric_stove_diff_1k.png', '49eeb4590e1b66f4b4bc23787932db25d2216ea851f219de42701cf5e38fe48b'),
    ],
    # Objaverse, at a pinned dataset commit. Poly Haven's sofas are antique styles YOLO calls "bed" or "bench",
    # and its carved Gothic bed was taken for people, horses and birds.
    "couch_starstarbird": [
        ('couch_starstarbird.glb', f'{OBJAVERSE}/glbs/000-072/d90e4353ae7047e0b4f37abc8bc58468.glb', '9e366736149477800f04c7cfaa7e3a2529ac04000a0336b6f4ab84b3e096c669'),
    ],
    "bed_tharadelamo": [
        ('bed_tharadelamo.glb', f'{OBJAVERSE}/glbs/000-099/fc9cbf532b6045969b9f7eeac339afd2.glb', '5d0682406b751601754be00da984acad525ca4234acdedf3bf6e61ce98438b40'),
    ],
}
# Models that aren't Poly Haven's CC0 ones: licence, author, page. CC-BY requires this credit.
CREDITS = {
    "couch_starstarbird": ("CC-BY 4.0", "StarStarbird",
                           "https://sketchfab.com/3d-models/couch-d90e4353ae7047e0b4f37abc8bc58468"),
    "bed_tharadelamo": ("CC-BY 4.0", "tharadelamo",
                        "https://sketchfab.com/3d-models/fc9cbf532b6045969b9f7eeac339afd2"),
}
COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _fetch(path, url, sha256):
    """Download `url` to `path` unless a file with the pinned SHA-256 is already there."""
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "go2-mujoco-sim"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{path.name}: download doesn't match its pinned checksum; refusing it")
    path.write_bytes(data)


def _accessor(gltf, blob, index):
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype, width = np.dtype(COMPONENTS[accessor["componentType"]]), WIDTHS[accessor["type"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    return np.ndarray((accessor["count"], width), dtype, blob, offset, (stride, dtype.itemsize)).copy()


def _local(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T          # glTF stores matrices column-major
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array(node.get("scale", (1, 1, 1)))
    matrix[:3, 3] = node.get("translation", (0, 0, 0))
    return matrix


def _load(folder, name):
    """The glTF JSON and its binary buffer: from name.gltf and its .bin, or from a single name.glb (a
    12-byte header, then a JSON chunk and a binary chunk, each with an 8-byte length-and-type header)."""
    glb = folder / f"{name}.glb"
    if glb.exists():
        data = glb.read_bytes()
        (json_length,) = struct.unpack_from("<I", data, 12)
        (bin_length,) = struct.unpack_from("<I", data, 20 + json_length)
        return json.loads(data[20:20 + json_length]), data[28 + json_length:28 + json_length + bin_length]
    gltf = json.loads((folder / f"{name}.gltf").read_text())
    return gltf, (folder / gltf["buffers"][0]["uri"]).read_bytes()


def convert(folder, name):
    """glTF -> one OBJ per colour texture (Z-up, centred, on the floor) and model.json describing them."""
    gltf, blob = _load(folder, name)
    groups = {}                                        # texture (or colour) -> [(positions, uvs, triangles)]

    def visit(index, parent):
        node = gltf["nodes"][index]
        world = parent @ _local(node)
        for primitive in gltf["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []:
            if primitive.get("mode", 4) != 4:
                continue                                # triangles only
            points = _accessor(gltf, blob, primitive["attributes"]["POSITION"]).astype(float)
            points = points @ world[:3, :3].T + world[:3, 3]
            uvs = (_accessor(gltf, blob, primitive["attributes"]["TEXCOORD_0"]).astype(float)
                   if "TEXCOORD_0" in primitive["attributes"] else np.zeros((len(points), 2)))
            triangles = _accessor(gltf, blob, primitive["indices"]).reshape(-1, 3).astype(np.int64)
            pbr = gltf["materials"][primitive["material"]].get("pbrMetallicRoughness", {}) if "material" in primitive else {}
            if "baseColorTexture" in pbr:
                uri = gltf["images"][gltf["textures"][pbr["baseColorTexture"]["index"]]["source"]]["uri"]
                key = ("texture", uri.rsplit(".", 1)[0] + ".png")
            else:
                key = ("rgba", tuple(round(c, 3) for c in pbr.get("baseColorFactor", (.8, .8, .8, 1))))
            groups.setdefault(key, []).append((points, uvs, triangles))
        for child in node.get("children", []):
            visit(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    z_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)       # glTF is Y-up, MuJoCo Z-up
    every = np.concatenate([p for parts in groups.values() for p, _, _ in parts]) @ z_up.T
    lo, hi = every.min(0), every.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    out = DEST / name
    described = []
    for i, ((kind, value), parts) in enumerate(groups.items()):
        lines, base = [], 1
        for points, uvs, triangles in parts:
            for x, y, z in points @ z_up.T + shift:
                lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
            for u, v in uvs:
                lines.append(f"vt {u:.5f} {1 - v:.5f}")             # glTF's v runs top-down, OBJ's bottom-up
            for a, b, c in triangles + base:
                lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            base += len(points)
        (out / f"part{i}.obj").write_text("\n".join(lines) + "\n")
        part = {"obj": f"part{i}.obj"}
        if kind == "texture":
            shutil.copyfile(folder / value, out / f"part{i}.png")
            part["texture"] = f"part{i}.png"
        else:
            part["rgba"] = list(value)
        described.append(part)
    return {"size": [round(float(v), 3) for v in hi - lo], "parts": described}


def main():
    for name, files in MODELS.items():
        described = DEST / name / "model.json"
        pinned = [sha256 for _, _, sha256 in files]
        if described.exists() and json.loads(described.read_text()).get("sha256") == pinned:
            info = json.loads(described.read_text())          # converted from exactly these files already
        else:
            source = DEST / name / "source"
            for path, url, sha256 in files:
                _fetch(source / path, url, sha256)
            info = convert(source, name)
            licence, author, page = CREDITS.get(name, ("CC0 1.0", "Poly Haven", f"https://polyhaven.com/a/{name}"))
            info.update(model=name, license=licence, author=author, url=page, sha256=pinned)
            described.write_text(json.dumps(info, indent=1))
            shutil.rmtree(source)                              # the originals aren't needed once converted
        size = sum(p.stat().st_size for p in (DEST / name).rglob("*") if p.is_file())
        print(f"{name:24s} {len(info['parts'])} part(s)  {info['size']} m  {size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
