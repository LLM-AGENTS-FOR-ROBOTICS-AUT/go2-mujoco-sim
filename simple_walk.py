import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("_deps/unitree_mujoco/unitree_robots/go2/scene.xml")
data = mujoco.MjData(model)

# Walking parameters
t = 0
dt = 0.01

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Simple walking pattern using sine waves
        for i in range(12):  # 12 actuators
            data.ctrl[i] = 0.5 * np.sin(t + i)

        mujoco.mj_step(model, data)
        viewer.sync()

        t += dt
        time.sleep(dt)
