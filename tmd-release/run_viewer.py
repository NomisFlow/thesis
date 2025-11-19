import time

import mujoco
import mujoco.viewer
import numpy as np

from ogbench.manipspace.envs.cube_env import CubeEnv

def main():
    # 1. Instantiate CubeEnv
    env = CubeEnv(env_type='single')

    # 2. Call env.reset() to initialize the environment
    # The reset method will compile the model and data
    obs, info = env.reset()

    # 3. Launch the passive viewer
    with env.passive_viewer() as viewer:
        # 4. Step the environment in a loop to keep the viewer updated
        while viewer.is_running():
            # Take a dummy action (e.g., zeros)
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)

            # If the episode terminates or truncates, reset the environment
            if terminated or truncated:
                obs, info = env.reset()

            # 5. Call env.sync_passive_viewer() to update the viewer
            viewer.sync()
            time.sleep(env.control_timestep())

if __name__ == '__main__':
    main()
