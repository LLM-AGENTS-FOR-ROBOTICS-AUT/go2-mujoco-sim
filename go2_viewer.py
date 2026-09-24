"""Shared Windows and macOS viewer for the official Unitree Go2 MuJoCo model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent
GO2_DIR = ROOT / "_deps" / "unitree_mujoco" / "unitree_robots" / "go2"


def standing_targets() -> dict[str, float]:
    return {
        "FR_hip_joint": 0.10,
        "FR_thigh_joint": 0.90,
        "FR_calf_joint": -1.80,
        "FL_hip_joint": -0.10,
        "FL_thigh_joint": 0.90,
        "FL_calf_joint": -1.80,
        "RR_hip_joint": 0.10,
        "RR_thigh_joint": 0.90,
        "RR_calf_joint": -1.80,
        "RL_hip_joint": -0.10,
        "RL_thigh_joint": 0.90,
        "RL_calf_joint": -1.80,
    }


def set_standing_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Go2 and Go2 EDU use the same simulated mechanics. EDU differences are in
    # the onboard computer and API access rather than the body geometry.
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    if model.nq >= 7:
        data.qpos[2] = 0.32
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)

    for joint_name, target in standing_targets().items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = target
    mujoco.mj_forward(model, data)


def load_scene(terrain: bool) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    scene = GO2_DIR / ("scene_terrain.xml" if terrain else "scene.xml")
    if not scene.exists():
        setup = "bash macos/setup.sh" if sys.platform == "darwin" else "windows/setup.ps1"
        raise FileNotFoundError(
            f"Go2 scene not found: {scene}\nRun {setup} first."
        )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    set_standing_pose(model, data)
    return model, data, scene


def apply_standing_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    targets = standing_targets()
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        joint_name = f"{actuator_name}_joint"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or joint_name not in targets:
            continue
        qpos_address = model.jnt_qposadr[joint_id]
        dof_address = model.jnt_dofadr[joint_id]
        kp, kd = ((28.0, 1.0) if "hip_joint" in joint_name else (35.0, 2.0))
        data.ctrl[actuator_id] = (
            kp * (targets[joint_name] - data.qpos[qpos_address])
            - kd * data.qvel[dof_address]
        )


def validate(model: mujoco.MjModel, data: mujoco.MjData, scene: Path) -> int:
    for _ in range(500):
        apply_standing_controller(model, data)
        mujoco.mj_step(model, data)
    if not 0.15 < data.qpos[2] < 0.6:
        raise RuntimeError(f"unexpected base height after test: {data.qpos[2]:.3f} m")
    print(f"MuJoCo {mujoco.__version__}")
    print(f"Loaded: {scene}")
    print(
        f"Go2 model OK: {model.nbody} bodies, {model.njnt} joints, "
        f"{model.nu} actuators"
    )
    print(f"Simulation check OK: t={data.time:.3f}s, base height={data.qpos[2]:.3f}m")
    return 0


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: Path,
    auto_close_seconds: float | None,
) -> int:
    import mujoco.viewer

    print(f"Opening {scene.name} with the Unitree Go2 EDU model.")
    print("Mouse: rotate/pan/zoom. Press Esc or close the window to exit.")
    started = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.25)
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        while viewer.is_running():
            step_started = time.monotonic()
            apply_standing_controller(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            if auto_close_seconds and time.monotonic() - started >= auto_close_seconds:
                break
            remaining = model.opt.timestep - (time.monotonic() - step_started)
            if remaining > 0:
                time.sleep(remaining)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Unitree Go2 in MuJoCo")
    parser.add_argument(
        "--terrain", action="store_true", help="use Unitree's rough-terrain scene"
    )
    parser.add_argument(
        "--validate", action="store_true", help="check the installation headlessly"
    )
    parser.add_argument("--auto-close-seconds", type=float, default=None)
    parser.add_argument("--walk", action="store_true", help="use the optional Go2 locomotion policy")
    parser.add_argument("--demo", action="store_true", help="run the walking demonstration (implies --walk)")
    parser.add_argument("--report", type=Path, default=ROOT / "_deps" / "locomotion-results.json",
                        help="JSON report path for --walk --validate")
    args = parser.parse_args()
    if (args.walk or args.demo) and args.terrain and args.validate:
        parser.error("The locomotion tracking suite uses level ground; omit --terrain")
    if args.auto_close_seconds is not None and (
        not 0 < args.auto_close_seconds < float("inf")
    ):
        parser.error("--auto-close-seconds must be a finite positive number")
    try:
        if args.walk or args.demo:
            try:
                from go2_locomotion import SimSportClient, run_viewer as run_walking_viewer
            except ImportError as exc:
                raise RuntimeError(
                    "Walking dependencies are missing. Run bash macos/setup.sh --locomotion "
                    "or Windows setup.ps1 -Locomotion."
                ) from exc
            scene = GO2_DIR / ("scene_terrain.xml" if args.terrain else "scene.xml")
            client = SimSportClient(scene)
            if args.validate:
                from validate_locomotion import validate as validate_walking
                return validate_walking(client, args.report)
            return run_walking_viewer(client, args.demo, args.auto_close_seconds)
        model, data, scene = load_scene(args.terrain)
        if args.validate:
            return validate(model, data, scene)
        return run_viewer(model, data, scene, args.auto_close_seconds)
    except Exception as exc:
        print(f"Could not start the Go2 simulator: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
