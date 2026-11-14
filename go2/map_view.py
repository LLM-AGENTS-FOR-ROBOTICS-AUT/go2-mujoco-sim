"""Bounded read-only map presentation; never acquires sensors or moves a robot."""
import base64
import math
import time
import numpy as np
from .adapters.base import AdapterError


def map_view(executor, max_pixels=384):
    if not isinstance(max_pixels, int) or not 32 <= max_pixels <= 512:
        raise ValueError("map view must have 32 to 512 pixels per side")
    rooms = executor.rooms()          # cached for a few seconds: splitting the map takes a few tenths of a second
    with executor.lock:
        adapter, grid = executor.adapter, executor.room_map
        current = executor.current
        route = list(current.detail.get("route", []))[:512] if current else []
        if current and not route and current.detail.get("destination"):
            goal = current.detail["destination"]
            route = [[goal["x_m"], goal["y_m"]]]
        places = executor.places.all()
        objects = executor.object_memory.snapshot()
    robot, error = None, None
    if adapter is not None and adapter.connected:
        try:
            state = adapter.state()
            robot = dict(x=state.x, y=state.y, yaw=state.yaw, age_s=max(0., state.age),
                         localization=state.extra.get("localization"))
        except AdapterError as problem:
            error = str(problem)
    else:
        error = "Robot disconnected"
    with grid._lock:
        cells = grid.grid.copy()
        origin = np.array([grid._min_col, grid._min_row])*grid.resolution
        resolution = grid.resolution
    # Keep the observed area plus the robot and saved annotations in view.
    known = np.argwhere(cells != -1)
    extra = [[p.x, p.y] for p in places]
    extra += [[p["x"], p["y"]] for p in objects]
    if robot:
        extra += [[robot["x"], robot["y"]]]
    if extra:
        positions = np.floor((np.asarray(extra)-origin)/resolution).astype(int)[:, ::-1]
        positions = np.clip(positions, 0, len(cells)-1)
        known = np.vstack([known, positions])
    if len(known):
        low = np.maximum(0, known.min(axis=0)-8)
        high = np.minimum(cells.shape, known.max(axis=0)+9)
    else:
        low, high = np.array([0, 0]), np.array(cells.shape)
    cells = cells[low[0]:high[0], low[1]:high[1]]
    origin += low[::-1]*resolution
    stride = max(1, math.ceil(max(cells.shape)/max_pixels))
    height, width = (math.ceil(v/stride) for v in cells.shape)
    # Pool conservatively: occupied wins over free, free wins over unknown.
    encoded = np.zeros((height*stride, width*stride), dtype=np.uint8)
    encoded[:cells.shape[0], :cells.shape[1]] = cells+1
    encoded = encoded.reshape(height, stride, width, stride).max(axis=(1, 3))
    return dict(frame="map", generated_at=time.time(), error=error,
                grid=dict(width=width, height=height, resolution_m=resolution*stride,
                          origin_m=origin.tolist(), encoding="base64-u8", values={"unknown": 0, "free": 1, "occupied": 2},
                          data=base64.b64encode(encoded.tobytes()).decode()),
                robot=robot, route=[[float(x) for x in p[:2]] for p in route],
                places=[dict(name=p.name, x=p.x, y=p.y, yaw=p.yaw) for p in places[:200]],
                places_total=len(places),
                rooms=[dict(name=r["name"], x=r["x"], y=r["y"]) for r in rooms[:50]], rooms_total=len(rooms),
                objects=[{k: p.get(k) for k in ("id", "label", "color", "x", "y", "state", "last_seen")}
                         for p in objects[-200:]], objects_total=len(objects))
