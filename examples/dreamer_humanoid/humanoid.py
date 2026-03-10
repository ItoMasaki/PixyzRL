# humanoid_headcam_force_gym.py
from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt


class HumanoidStandingEnv(gym.Env):
    """
    Minimal Genesis humanoid Gym env with:
      - head-mounted camera
      - velocity control (actions in [-1,1] -> target joint velocities)
      - reward: alive bonus + upright - motion penalty
      - gravity and mass scaling supported via genesis API
    """

    def __init__(self, healthy_reward: float = 5.0, healthy_range: tuple[float, float] = (1.0, 2.0)):
        super().__init__()

        self.healthy_reward = healthy_reward
        self.healthy_range = healthy_range

        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            show_viewer=True,
        )

        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.humanoid = self.scene.add_entity(gs.morphs.MJCF(file="humanoid.xml", scale=1.0, pos=(0, 0, 0)))
        self.head_camera = self.scene.add_camera()

        self.scene.build()

        self.humanoid_joints_num = len(self.humanoid.joints)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.humanoid_joints_num,),
            dtype=np.float32,
        )

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.scene.reset()
        obs = {}
        info = {}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # Calculate camera pose
        self.calculate_camera_pose()

        # Apply action (target joint velocities)
        self.humanoid.set_dofs_velocity(np.zeros(self.humanoid_joints_num), range(self.humanoid_joints_num))

        # Get current state for observation
        color, _, _, _ = self.get_head_camera_image()
        plt.cla()
        plt.imshow(color)
        plt.pause(0.001)
        position = self.get_dofs_position()
        velocity = self.get_dofs_velocity()
        contact_mask = self.get_contact_info()

        # Get current state for reward calculation
        qpos = self.get_qpos()

        # Forward simulation
        self.scene.step()

        # Return observation, reward, done, info

    def get_contact_info(self):
        contact_mask = np.zeros(len(self.scene.rigid_solver.geoms), dtype=bool)
        for geom in self.humanoid.get_contacts().get("geom_b")[self.humanoid.get_contacts().get("geom_a") == 0]:
            contact_mask[self.scene.rigid_solver.geoms[geom].idx] = True

        return contact_mask

    def get_dofs_position(self):
        return self.humanoid.get_dofs_position()

    def get_dofs_velocity(self):
        return self.humanoid.get_dofs_velocity()

    def get_qpos(self):
        return self.humanoid.get_qpos()

    def get_head_camera_image(self):
        return self.head_camera.render()

    def calculate_camera_pose(self):
        print(dir(self.humanoid))
        head_link = self.humanoid.get_link("head")

        # --- 毎ステップ ---
        head_pos = head_link.get_pos()  # (3,)
        head_quat = head_link.get_quat()  # (4,)

        # 頭部のワールド変換
        T_head = self.to_matrix(head_pos, head_quat)

        # カメラを頭の前方10cm・上方5cmに固定したい場合
        offset = np.eye(4)
        offset[:3, 3] = [0.1, 0.0, 0.05]  # ローカルオフセットだけ

        rot90 = self.quat_from_axis_angle([0, 1, 0], -np.pi / 2)

        T_rot90 = self.to_matrix([0, 0, 0], rot90)

        # カメラのローカル変換 = 平行移動 @ 回転オフセット
        T_cam_local = offset @ T_rot90

        # ワールド変換に適用
        T_cam = T_head @ T_cam_local

        # カメラに pose を設定
        self.head_camera.set_pose(
            transform=T_cam,
        )

    def to_matrix(self, pos, quat):
        """pos=(x,y,z), quat=(w,x,y,z) → 4x4変換行列"""
        w, x, y, z = quat
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def quat_from_axis_angle(self, axis, angle):
        """axis=(x,y,z), angle[rad] から四元数(w,x,y,z)を作成"""
        axis = np.array(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        s = np.sin(angle / 2)
        return np.array([np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


if __name__ == "__main__":
    env = HumanoidStandingEnv()

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    for i in range(1000):
        #     # action = env.action_space.sample()
        #     # obs, reward, terminated, truncated, info = env.step(action)
        env.step(0)
    #     done = False
    #     total_reward += 0
    #     print(f"Step reward: {0}, Total reward: {total_reward}")
