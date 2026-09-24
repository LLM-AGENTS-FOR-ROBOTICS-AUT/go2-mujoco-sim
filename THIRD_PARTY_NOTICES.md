# Third-party components

The setup scripts download, but do not redistribute in this repository:

- Unitree MuJoCo: <https://github.com/unitreerobotics/unitree_mujoco>
- Unitree SDK2: <https://github.com/unitreerobotics/unitree_sdk2>
- MuJoCo: <https://github.com/google-deepmind/mujoco>
- uv (macOS Python installer): <https://github.com/astral-sh/uv>
- Python builds used by uv: <https://github.com/astral-sh/python-build-standalone>
- PyTorch (CPU policy inference): <https://github.com/pytorch/pytorch>
- Go2 Walk These Ways weights and configuration:
  <https://github.com/lupinjia/go2_deploy>

The optional walking setup downloads `models/wtw_model.pt` from go2_deploy
commit `2ee4388672fa121eb6f827984dfdeb16e415fabe`. Its SHA-256 is recorded in
`policy_assets.py`. The weights are downloaded into `_deps`; they are not
redistributed in this repository. The upstream repository has no top-level
license file at that revision.

The local adapter follows the observation, history, action, and gait conventions
in [WTWController](https://github.com/lupinjia/go2_deploy/blob/2ee4388672fa121eb6f827984dfdeb16e415fabe/include/user_controller.hpp).
It adds simulated pose feedback, a neutral gait setting, bounded PD torques,
and a stop-to-stand transition.

The optional comparison harness also downloads models/configuration from:

- rl_sar (Apache-2.0), including HIMLoco: <https://github.com/fan-ziqi/rl_sar>
- go2_deploy_python: <https://github.com/lupinjia/go2_deploy_python>

Exact comparison URLs, revisions and checksums are in `experiments/sources.json`.
The earlier rejected robot_lab baseline used the ONNX model from Efferent
(Apache-2.0): <https://github.com/Eximius-Labs/efferent>. It is not installed by
the final walking setup.

Each downloaded project remains subject to its own license. Review those
licenses before redistribution or commercial use.
