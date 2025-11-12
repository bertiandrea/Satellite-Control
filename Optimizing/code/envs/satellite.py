# satellite.py
from code.envs.vec_task import VecTask

import isaacgym #BugFix
import torch
from isaacgym import gymutil, gymtorch, gymapi

from pathlib import Path
import numpy as np

class Satellite(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.env_spacing =           cfg["env"].get('envSpacing', 0.0)                       # meters
        self.asset_name =            cfg["env"]["asset"].get('assetName', 'satellite')
        self.asset_root =            cfg["env"]["asset"].get('assetRoot', str(Path(__file__).resolve().parent.parent))
        self.asset_file =            cfg["env"]["asset"].get('assetFileName', 'satellite.urdf')

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)


    def create_sim(self) -> None:
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params) 
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            ###################################################
            asset_init_pos_p = [0, 0, 0]
            asset_init_pos_r = np.random.randn(4)
            asset_init_pos_r /= np.linalg.norm(asset_init_pos_r)
            ###################################################
            self.create_actor(i, env, self.asset, asset_init_pos_p, asset_init_pos_r, 1, self.asset_name)
            ###################################################
            self.envs.append(env)

    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        return asset
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        actor_handle =  self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)
        return actor_handle
    
    def pre_physics_step(self, actions):
        return
    def post_physics_step(self):
        return
    
    def close(self):
        print("Closing Satellite environment...")
        del self.env_spacing, self.asset_name, self.asset_root, self.asset_file, self.asset
        torch.cuda.empty_cache()  # Empty GPU cache
        super().close()
