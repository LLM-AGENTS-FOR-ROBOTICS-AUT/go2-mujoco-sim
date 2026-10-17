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

## LLM command console (`--agent` setup)

- go2-ft-04 is a fine-tuned derivative of Google DeepMind's Gemma 4 E2B. The official model card
  identifies Apache 2.0: <https://ai.google.dev/gemma/docs/core/model_card_4>. The weights are not in this
  repository: the console downloads them from <https://huggingface.co/SageStack01/go2-ft-04> (Apache 2.0)
  on its first start, and `models/catalog.json` pins the exact upload and its SHA-256.
  No production approval is granted.
- Ollama runs the model: <https://github.com/ollama/ollama> (installed separately by each user).
- The simulated room's household objects come from **Google Scanned Objects** by Google LLC,
  **CC-BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>), via
  <https://github.com/kevinzakka/mujoco_scanned_objects> (MJCF files MIT) at commit `6ff8d275`.
  `simulation/fetch_gso.py` downloads them into `_deps/gso`; the meshes are used unmodified.
- Sensor mount positions for the simulated L1 LiDAR and front camera follow Unitree's `go2_description`
  URDF (unitree_ros, BSD 3-Clause). The room itself is original.
- Speech recognition: OpenAI's Whisper large-v3-turbo weights (MIT), downloaded from Hugging Face on first
  use at pinned revisions: `mlx-community/whisper-large-v3-turbo` for mlx-whisper (MIT) on Apple Silicon,
  `dropbox-dash/faster-whisper-large-v3-turbo` for faster-whisper (MIT) elsewhere. The original page's
  5-second voice button uses faster-whisper with Whisper `base` (MIT).
- Go2's voice: Kokoro-82M by hexgrad (Apache 2.0, weights and voices downloaded from Hugging Face on first
  use at a pinned revision) through the `kokoro` package (Apache 2.0). Its pronunciation uses `misaki`
  (Apache 2.0) with spaCy and `en_core_web_sm` (MIT), and for unusual words `phonemizer-fork` and eSpeak NG
  (**GPL-3.0**), installed as dependencies; this repository doesn't include any of them.
- Optional, for the real robot only (`requirements-robot.txt`, `requirements-vision.txt`):
  `unitree_sdk2py` (BSD 3-Clause), CycloneDDS (EPL-2.0 / EDL-1.0), OpenCV (Apache 2.0) and
  `ultralytics` with YOLO weights (**AGPL-3.0**: using it in a distributed or networked product has
  copyleft obligations; consider a differently licensed detector if that matters).

Each downloaded project remains subject to its own license. Review those
licenses before redistribution or commercial use.
