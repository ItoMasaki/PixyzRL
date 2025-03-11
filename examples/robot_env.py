import random
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from gymnasium import spaces
from numpy.typing import NDArray


class BipedalRobotEnv(gym.Env[Any, Any]):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, render_mode: str = "human"):
        super().__init__()

        self.render_mode = render_mode

        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pd.getDataPath())
        self._setup_simulation()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(p.getNumJoints(self.botId),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(p.getNumJoints(self.botId) + 1,), dtype=np.float32)

    def _setup_simulation(self) -> None:
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(1e-3, physicsClientId=self.client_id)

        # Create Terrain
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        heightfieldData = [random.uniform(0, 0.1) for _ in range(numHeightfieldRows * numHeightfieldColumns)]
        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[0.1, 0.1, 1], heightfieldTextureScaling=numHeightfieldRows, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns, physicsClientId=self.client_id)
        self.terrain = p.createMultiBody(0, terrainShape, physicsClientId=self.client_id)
        p.resetBasePositionAndOrientation(self.terrain, [0, 0, -0.3], [0, 0, 0, 1], physicsClientId=self.client_id)

        # Load Robot
        cubeStartPos = [0, 0, 0.57]
        cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0, 0])
        self.botId = p.loadURDF("/Users/itomasaki/Desktop/PixyzRL/examples/mybot.urdf", cubeStartPos, cubeStartOrientation, physicsClientId=self.client_id)

        for joint in range(p.getNumJoints(self.botId)):
            p.setJointMotorControl2(self.botId, joint, p.POSITION_CONTROL, force=1.0, physicsClientId=self.client_id)

    def step(self, action: NDArray[np.float32]) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        for joint_idx, joint_action in enumerate(action):
            p.setJointMotorControl2(self.botId, joint_idx, p.POSITION_CONTROL, force=1.0, targetPosition=joint_action, physicsClientId=self.client_id)

        p.stepSimulation(physicsClientId=self.client_id)

        body_height = p.getLinkState(self.botId, 0, physicsClientId=self.client_id)[0][2]
        obs = self._get_observation()
        reward = 1.0 if (body_height > 0.425) and (body_height < 0.475) else 0  # Reward for staying high (simplified)
        collision = p.getContactPoints(self.botId, self.terrain, -1, -1, physicsClientId=self.client_id)
        done = len(collision) > 0
        info = {}

        return obs, reward, done, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        p.resetBasePositionAndOrientation(self.botId, [0, 0, 0.57], [0, 0, 0, 1], physicsClientId=self.client_id)
        for joint_idx in range(p.getNumJoints(self.botId)):
            p.resetJointState(self.botId, joint_idx, 1, physicsClientId=self.client_id)

        return self._get_observation(), {}

    def _get_observation(self) -> NDArray[Any]:
        joint_positions = [p.getJointState(self.botId, i, physicsClientId=self.client_id)[0] for i in range(p.getNumJoints(self.botId))]
        body_height = p.getLinkState(self.botId, 0, physicsClientId=self.client_id)[0][2]
        return np.array([*joint_positions, body_height], dtype=np.float32)

    def render(self, mode: str = "human") -> Any:
        if mode == "rgb_array":
            width, height, _, _, _, _ = p.getCameraImage(320, 240, physicsClientId=self.client_id)
            return np.array(width).reshape((240, 320, 3))
        return None

    def close(self) -> None:
        p.disconnect(self.client_id)


if __name__ == "__main__":
    env = BipedalRobotEnv()
    obs = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        action = -np.ones(19)
        obs, reward, done, truncated, _ = env.step(action)
        print(reward)
        if done:
            env.reset()
    env.close()
