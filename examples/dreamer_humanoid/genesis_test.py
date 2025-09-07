# humanoid_headcam_force_gym.py
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import genesis as gs


class HumanoidHeadCamForceEnv(gym.Env):
    """
    Minimal Genesis humanoid Gym env (force control):
      - Head-mounted camera provides image observation
      - Action: joint torques in [-1, 1], scaled by force_scale
      - Termination ONLY (height or time)
      - Reward via self.compute_reward(q, qd):
          * + alive_bonus (per step survival reward)
          * + upright (0..1): stand upright gets positive reward
          * - joint motion penalty proportional to ||qd||^2
    """

    metadata = {"render_modes": ["human", "rgb_array", "rgb_array_world"], "render_fps": 60}

    def __init__(
        self,
        humanoid_xml: str = "xml/humanoid.xml",
        backend: str = "gpu",
        head_cam_res: Tuple[int, int] = (320, 240),
        head_cam_fov: float = 80.0,
        head_link_name: str = "head",
        torso_link_name: str = "torso",
        head_cam_pos: Tuple[float, float, float] = (0.10, 0.0, 0.0),
        head_cam_xyaxes: Tuple[float, float, float, float, float, float] = (0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
        world_cam_enabled: bool = True,
        world_cam_res: Tuple[int, int] = (800, 600),
        world_cam_fov: float = 70.0,
        world_cam_pos: Tuple[float, float, float] = (3.0, 0.0, 1.8),
        world_cam_xyaxes: Tuple[float, float, float, float, float, float] = (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        world_cam_capture: bool = False,
        show_viewer: bool = True,
        render_mode: Optional[str] = "human",
        # control / episode
        force_scale: float = 50.0,
        max_steps_per_episode: int = 1000,
        z_head_min: float = 1.0,
        z_torso_min: float = 0.75,
        # ==== reward weights ====
        alive_bonus: float = 3.0,  # 生存報酬（毎ステップ）
        w_upright: float = 1.0,     # 直立のご褒美
        w_motion: float = 0.01,     # 動きのペナルティ係数
    ):
        super().__init__()
        self.humanoid_xml = humanoid_xml
        self.backend = backend

        self.head_cam_res = head_cam_res
        self.head_cam_fov = head_cam_fov
        self.head_link_name = head_link_name
        self.torso_link_name = torso_link_name
        self.head_cam_pos = head_cam_pos
        self.head_cam_xyaxes = head_cam_xyaxes

        self.world_cam_enabled = world_cam_enabled
        self.world_cam_res = world_cam_res
        self.world_cam_fov = world_cam_fov
        self.world_cam_pos = world_cam_pos
        self.world_cam_xyaxes = world_cam_xyaxes
        self.world_cam_capture = world_cam_capture
        self.show_viewer = bool(show_viewer)
        self.render_mode = render_mode

        self.force_scale = float(force_scale)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.z_head_min = float(z_head_min)
        self.z_torso_min = float(z_torso_min)

        self.alive_bonus = float(alive_bonus)
        self.w_upright = float(w_upright)
        self.w_motion = float(w_motion)

        # Genesis init
        try:
            gs.assert_initialized()
        except Exception:
            gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

        self._build_scene()
        self._setup_spaces()
        self._t = 0

    # ---------- build ----------
    def _build_scene(self):
        self.scene = gs.Scene(show_viewer=self.show_viewer)
        self.humanoid = self.scene.add_entity(gs.morphs.MJCF(file=self.humanoid_xml))

        self.world_cam = None
        if self.world_cam_enabled:
            self.world_cam = self.scene.add_camera(
                model="pinhole", res=self.world_cam_res, fov=self.world_cam_fov, GUI=True
            )
        self.head_cam = self.scene.add_camera(
            model="pinhole", res=self.head_cam_res, fov=self.head_cam_fov, GUI=False
        )

        self.scene.build()

        self._head = self.humanoid.get_link(self.head_link_name)
        self._torso = self.humanoid.get_link(self.torso_link_name)

        # attach head camera
        T_head = self._xyaxes_to_T(self.head_cam_pos, self.head_cam_xyaxes)
        self.head_cam.attach(self._head, T_head)
        self.head_cam.move_to_attach()

        # place world cam
        if self.world_cam is not None:
            T_w = self._xyaxes_to_T(self.world_cam_pos, self.world_cam_xyaxes)
            if hasattr(self.world_cam, "set_transform"):
                self.world_cam.set_transform(T_w)
            elif hasattr(self.world_cam, "set_pose"):
                self.world_cam.set_pose(pos=tuple(self.world_cam_pos))

        # dofs
        dof_idx = []
        for j in self.humanoid.joints:
            if hasattr(j, "dofs_idx"):
                dof_idx.extend(list(j.dofs_idx))
            elif hasattr(j, "dof_idx"):
                dof_idx.append(int(j.dof_idx))
        self.dofs_idx = sorted(set(int(i) for i in dof_idx))
        self.ndof = len(self.dofs_idx)

    def _setup_spaces(self):
        H, W = self.head_cam_res[1], self.head_cam_res[0]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(H, W, 3), dtype=np.uint8),
                "pose": spaces.Box(-np.inf, np.inf, shape=(self.ndof * 2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.ndof,), dtype=np.float32)

    # ---------- utils ----------
    @staticmethod
    def _xyaxes_to_T(pos_xyz, xyaxes6):
        x = np.array(xyaxes6[:3], dtype=float); x /= (np.linalg.norm(x) + 1e-12)
        y = np.array(xyaxes6[3:], dtype=float); y /= (np.linalg.norm(y) + 1e-12)
        z = np.cross(x, y)
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.stack([x, y, z], axis=1)
        T[:3, 3] = np.array(pos_xyz, dtype=float)
        return T

    def _head_rgb(self):
        rgb, _, _, _ = self.head_cam.render(rgb=True)
        return rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)

    def _terminated_by_height(self) -> bool:
        head_z = float(self._head.get_pos()[2])
        torso_z = float(self._torso.get_pos()[2])
        return (head_z < self.z_head_min) or (torso_z < self.z_torso_min)

    def _obs(self) -> Dict[str, Any]:
        img = self._head_rgb()
        # 必須：detach().cpu().numpy() を維持
        q = self.humanoid.get_dofs_position(self.dofs_idx).detach().cpu().numpy().astype(np.float32)
        qd = self.humanoid.get_dofs_velocity(self.dofs_idx).detach().cpu().numpy().astype(np.float32)
        return {"image": img, "pose": np.concatenate([q, qd])}

    # ---------- reward ----------
    def compute_reward(self, pose: np.ndarray) -> float:
        """
        Survival + Upright - Motion:
          reward = alive_bonus + w_upright * upright - w_motion * ||qd||^2
        upright is computed from torso quaternion (alignment of body +Z with world +Z).
        """
        # Upright in [0,1] from torso quaternion (w,x,y,z):
        quat = self._torso.get_quat().detach().cpu().numpy().astype(np.float32)
        n = float(np.linalg.norm(quat))
        if n > 0:
            quat /= n
        x, y = float(quat[1]), float(quat[2])
        upright = float(np.clip(1.0 - 2.0 * (x * x + y * y), 0.0, 1.0))

        # Motion penalty
        motion = float(np.sum(
            pose[self.ndof:].astype(np.float32) ** 2))

        return self.alive_bonus + self.w_upright * upright - self.w_motion * motion

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._t = 0
        try:
            if hasattr(self.scene, "reset"):
                self.scene.reset()
        except Exception:
            self._build_scene()
            self._setup_spaces()
        self.scene.step()
        self.head_cam.move_to_attach()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.ndof,):
            raise ValueError(f"action shape {action.shape} != ({self.ndof},)")
        tau = self.force_scale * np.clip(action, -1.0, 1.0)
        self.humanoid.control_dofs_force(tau, self.dofs_idx)

        self.scene.step()
        self.head_cam.move_to_attach()
        self._t += 1

        obs = self._obs()
        terminated = self._terminated_by_height()
        truncated = (self._t >= self.max_steps_per_episode)

        reward = self.compute_reward(obs["pose"])

        info = {
            "head_z": float(self._head.get_pos().detach().cpu().numpy()[2]),
            "torso_z": float(self._torso.get_pos().detach().cpu().numpy()[2]),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._head_rgb()

    def close(self):
        pass


if __name__ == "__main__":
    env = HumanoidHeadCamForceEnv(
        humanoid_xml="xml/humanoid.xml",
        backend="cpu",
        show_viewer=True,
        force_scale=50.0,
        alive_bonus=0.01,  # 生存報酬
        w_upright=1.0,
        w_motion=0.01,
    )
    obs, _ = env.reset()
    for t in range(500):
        a = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(a)
        print(f"[{t+1}] reward={rew:.3f}, head_z={info['head_z']:.3f}, torso_z={info['torso_z']:.3f}")
        if term or trunc:
            break
    env.close()
