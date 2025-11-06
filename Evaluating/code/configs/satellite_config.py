# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

NUM_ENVS = 4096
TIMESTEPS = 100000
HEADLESS = False
DEBUG_ARROWS = True
LOG_TRAJECTORIES = True

CONFIG = {
    # --- seed & devices ----------------------------------------------------
    "set_seed": True,
    "seed": 42,

    "profile": False,

    "physics_engine": "physx",

    "rl_device": "cuda:0",
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": HEADLESS,
    "virtual_screen_capture": False,
    "force_render": False,

    # --- env section -------------------------------------------------------
    "env": {
        "numEnvs": NUM_ENVS,
        "numObservations": 15, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3)
        "numStates": 18, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3) + satellite_angvels (3)
        "numActions": 3,
       
        "clipActions": 1.0,
        "clipObservations": 10.0,

        "max_episode_length": 333.0,

        "envSpacing": 3.0,
        "torque_scale": 100.0,
        "debug_arrows": DEBUG_ARROWS,
        "debug_prints": False,
        "discretize_starting_pos": True,
        "log_trajectories": LOG_TRAJECTORIES,

        "asset": {
            "assetRoot": str(Path(__file__).resolve().parent.parent),
            "assetFileName": "satellite.urdf",
            "assetName": "satellite",
        },
    },

    # --- sim section -------------------------------------------------------
    "sim": {
        "dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, 0.0],
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "substeps": 2,

        "physx": {
            "use_gpu": True,
        },
    },

    # --- RL / PPO hyper-params --------------------------------------------
    "rl": {
        "PPO": {
            "num_envs": NUM_ENVS,
            
            "experiment": {
                "write_interval": "auto",
                "checkpoint_interval": "auto",
                "directory": "./runs",
                "wandb": False,
            },
        },
        "trainer": {
            "timesteps": TIMESTEPS,
            "disable_progressbar": False,
            "headless": HEADLESS,
            "stochastic_evaluation": False,
        },
    },
    # --- logging -----------------------------------------------------------
    "log_reward": {
        "log_reward": True,
        "log_reward_interval": 100,  # steps
    },
    # --- CAPS --------------------------------------------------------------
    "CAPS": {
        "enabled": False,
        "lambda_temporal_smoothness": 0.1,  # λ_t
        "lambda_spatial_smoothness": 0.1,   # λ_s
        "noise_std": 0.5,                   # σ
    },
    # --- explosion ---------------------------------------------------------
    "explosion": {
        "enabled": False,
        "explosion_time": 3,  # seconds
    },
    # --- asteroid ----------------------------------------------------------
    "asteroid": {
        "enabled": False,
        "object_mass": 0.0,  # kg
        "object_mass_std": 0.0,  # kg
        "object_mass_time": 120,  # seconds
    },
    # --- randomize masses --------------------------------------------------
    "randomize_masses": {
        "enabled": False,
        "mass_std": 5,
    }
}