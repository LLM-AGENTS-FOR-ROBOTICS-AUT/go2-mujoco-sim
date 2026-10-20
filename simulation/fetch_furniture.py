"""Download the realistic furniture used in the houses (house.py) and convert it for MuJoCo (~50 MB).

Source: Poly Haven (polyhaven.com), CC0: free for any use, no attribution required (we credit it anyway in
THIRD_PARTY_NOTICES.md). Every file is pinned by URL and MD5 (the checksum Poly Haven publishes); a file
that doesn't match is refused. Each glTF model becomes one OBJ mesh per texture, turned Z-up, centred on
x, y = 0 and standing on z = 0, plus model.json with its size. Without these files the houses use plain
shapes instead. The setup scripts run this with --agent / -Agent:

  .venv/bin/python simulation/fetch_furniture.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

DEST = Path(__file__).resolve().parents[1] / "_deps/furniture"
# model -> [(file, url, md5)]: the 1k glTF, its geometry and its colour textures (as PNG, which MuJoCo loads)
MODELS = {
    "dining_chair_02": [
        ('dining_chair_02.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/dining_chair_02/dining_chair_02_1k.gltf', '5aff75016e0943e84161de89dcc3ceb8'),
        ('dining_chair_02.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/dining_chair_02/dining_chair_02.bin', 'b7f8a91c0a5249d02de7885f3b242c4e'),
        ('textures/dining_chair_02_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/dining_chair_02/dining_chair_02_diff_1k.png', '1076dcfaa4d061177f180b97480c7938'),
    ],
    "painted_wooden_chair_01": [
        ('painted_wooden_chair_01.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/painted_wooden_chair_01/painted_wooden_chair_01_1k.gltf', '28e739db9799cdadf820ead2b486d463'),
        ('painted_wooden_chair_01.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/painted_wooden_chair_01/painted_wooden_chair_01.bin', '7fd011acaef4730770108a7932191768'),
        ('textures/painted_wooden_chair_01_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/painted_wooden_chair_01/painted_wooden_chair_01_diff_1k.png', 'cc3035abae2c6929e12b7fe17c74676c'),
    ],
    "potted_plant_02": [
        ('potted_plant_02.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/potted_plant_02/potted_plant_02_1k.gltf', '9999d7e6c51018a85551840e1bdae656'),
        ('potted_plant_02.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/potted_plant_02/potted_plant_02.bin', 'cdbd4619b2596c9c469ad34676085fa8'),
        ('textures/potted_plant_02_leaves_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_02/potted_plant_02_leaves_diff_1k.png', 'e40343a5d93db54caba5469b59cff76a'),
        ('textures/potted_plant_02_pot_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_02/potted_plant_02_pot_diff_1k.png', '550416d1c00b0ce2c8eba7de8aa5975d'),
    ],
    "potted_plant_01": [
        ('potted_plant_01.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/potted_plant_01/potted_plant_01_1k.gltf', '199e9e3d68f5447142db4066d60ae3f0'),
        ('potted_plant_01.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/potted_plant_01/potted_plant_01.bin', '18377bd8d99b20a8d5fc5cf03cbf94b2'),
        ('textures/potted_plant_01_leaves_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_01/potted_plant_01_leaves_diff_1k.png', 'a1e100797a055cb55dbbff2aa9095aaf'),
        ('textures/potted_plant_01_pot_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/potted_plant_01/potted_plant_01_pot_diff_1k.png', '83a8851050896c27c531863b02c8894a'),
    ],
    "Television_01": [
        ('Television_01.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/Television_01/Television_01_1k.gltf', '9456aa7effd3cb41a0e753d74293b1c3'),
        ('Television_01.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/Television_01/Television_01.bin', 'd28775c3746a6e7b0afa5fd217acbc44'),
        ('textures/Television_01_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/Television_01/Television_01_diff_1k.png', 'baab6cbd6c6a3f34c2044d32222bd4bf'),
    ],
    "GothicBed_01": [
        ('GothicBed_01.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/GothicBed_01/GothicBed_01_1k.gltf', '0635fcafb5d406ed9b10655a91f1eabb'),
        ('GothicBed_01.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/GothicBed_01/GothicBed_01.bin', '6311b36b274a2062e20083631acddd18'),
        ('textures/GothicBed_01_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/GothicBed_01/GothicBed_01_diff_1k.png', '52e9484fe23fedac7d3555dcab7fdf3a'),
    ],
    "electric_stove": [
        ('electric_stove.gltf', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/electric_stove/electric_stove_1k.gltf', '1ec96abe416e022f792ace742d33947a'),
        ('electric_stove.bin', 'https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/electric_stove/electric_stove.bin', '9846b43c4af868f79a95635fb9bd354d'),
        ('textures/electric_stove_diff_1k.png', 'https://dl.polyhaven.org/file/ph-assets/Models/png/1k/electric_stove/electric_stove_diff_1k.png', '272ec5a130b3fc231264a38e12e068db'),
    ],
}
COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _fetch(path, url, md5):
    """Download `url` to `path` unless a file with the pinned MD5 is already there."""
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == md5:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "go2-mujoco-sim"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if hashlib.md5(data).hexdigest() != md5:
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


def convert(folder, name):
    """glTF -> one OBJ per colour texture (Z-up, centred, on the floor) and model.json describing them."""
    gltf = json.loads((folder / f"{name}.gltf").read_text())
    blob = (folder / gltf["buffers"][0]["uri"]).read_bytes()
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
        pinned = [md5 for _, _, md5 in files]
        if described.exists() and json.loads(described.read_text()).get("md5") == pinned:
            info = json.loads(described.read_text())          # converted from exactly these files already
        else:
            source = DEST / name / "source"
            for path, url, md5 in files:
                _fetch(source / path, url, md5)
            info = convert(source, name)
            info.update(model=name, license="CC0 1.0 (Poly Haven)", url=f"https://polyhaven.com/a/{name}", md5=pinned)
            described.write_text(json.dumps(info, indent=1))
            shutil.rmtree(source)                              # the originals aren't needed once converted
        size = sum(p.stat().st_size for p in (DEST / name).rglob("*") if p.is_file())
        print(f"{name:24s} {len(info['parts'])} part(s)  {info['size']} m  {size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
