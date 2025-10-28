# optimize.py

from code.configs.satellite_config import CONFIG
from code.envs.satellite import Satellite
from code.models.custom_model import Shared
from code.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper
from code.rewards.satellite_reward import (
    SimpleReward,
    ReductionReward,
)

import isaacgym #BugFix
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from code.trainer.trainer import Trainer # Custom Trainer
from skrl.utils import set_seed

import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Optimization imports
import os
import gc
import json
import psutil
import optuna
from datetime import datetime
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
TENSORBOARD_TAG = "Reward / Instantaneous reward (mean)"
N_TRIALS = 1000
# ──────────────────────────────────────────────────────────────────────────────

REWARD_MAP = {
    "simple": SimpleReward,
    "reduction": ReductionReward,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training con reward function selezionabile")
    parser.add_argument(
        "--reward-fn",
        choices=list(REWARD_MAP.keys()),
        default="test",
        help="Which RewardFunction?"
    )
    return parser.parse_args()

def sample_ppo_params(trial: optuna.Trial):
    return {
        "rollouts": trial.suggest_categorical("rollouts", [16, 32, 64, 128]),
        "learning_epochs": trial.suggest_categorical("learning_epochs", [8, 16]),
        "mini_batches": trial.suggest_categorical("mini_batches", [2, 4, 8, 16]),
        #"discount_factor": trial.suggest_float("discount_factor", 0.90, 0.999),
        #"lambda":          trial.suggest_float("lambda", 0.90,   0.999),
        #"learning_rate":  trial.suggest_float("learning_rate", 1e-5, 1e-2),
        #"grad_norm_clip": trial.suggest_float("grad_norm_clip", 0.1, 1.0),
        #"ratio_clip":   trial.suggest_float("ratio_clip", 0.1, 0.3),
        #"value_clip": trial.suggest_float("value_clip", 0.1, 0.3),
        #"clip_predicted_values": trial.suggest_categorical("clip_predicted_values", [True, False]),
        #"entropy_loss_scale": trial.suggest_float("entropy_loss_scale", 0.0, 0.05),
        #"value_loss_scale": trial.suggest_float("value_loss_scale", 0.5, 2.0),
    }

def print_memory_usage(tag=""):
    """Stampa uso memoria GPU e CPU per debug di leak."""
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_mem = process.memory_info().rss / 1024**2  # in MB
    print(f"\n[MEMORY] {tag}")
    print(f"  CPU RSS: {cpu_mem:.2f} MB")
    mem_alloc = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"  GPU allocated: {mem_alloc:.2f} MB | reserved: {mem_reserved:.2f} MB")
    print("#" * len("[MEMORY] {tag}"))

def list_active_tensors(tag=""):
    """Stampa tutti i tensori ancora attivi su GPU."""
    tensors = []
    objs = gc.get_objects()
    for obj in objs:
        try:
            if torch.is_tensor(obj):
                tensors.append(obj)
                continue
            data_attr = getattr(obj, "data", None)
            if torch.is_tensor(data_attr):
                tensors.append(data_attr)
                continue
        except Exception:
            continue

    total_mem = 0.0
    cuda_tensors = 0
    for t in tensors:
        try:
            if t.is_cuda:
                mem = t.numel() * t.element_size() / 1024**2
                total_mem += mem
                cuda_tensors += 1
        except Exception:
            pass
    
    print(f"\n[DEBUG - ACTIVE TENSORS] {tag}")
    print(f"  Num tensors: {len(tensors)}")
    print(f"  Num CUDA tensors: {cuda_tensors}")
    print(f"  Total CUDA tensor memory: {total_mem:.2f} MB")
    print("-" * 60)
    return tensors

def objective(trial: optuna.Trial) -> float:
    print_memory_usage("#### BEFORE TRIAL START ####")  # Monitor memory before trial
    list_active_tensors("#### BEFORE TRIAL START ####")

    env = Satellite(
        cfg=CONFIG,
        rl_device=CONFIG["rl_device"],
        sim_device=CONFIG["sim_device"],
        graphics_device_id=CONFIG["graphics_device_id"],
        headless=CONFIG["headless"],
        virtual_screen_capture=CONFIG["virtual_screen_capture"],
        force_render= CONFIG["force_render"],
        reward_fn=REWARD_MAP[args.reward_fn](CONFIG["log_reward"]["log_reward"], CONFIG["log_reward"]["log_reward_interval"])
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=CONFIG["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Shared(env.state_space, env.action_space, env.device)
    models["value"] = models["policy"]  # Shared model for policy and value
   
    CONFIG["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    CONFIG["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update(CONFIG["rl"]["PPO"])

    hp = sample_ppo_params(trial)
    cfg_ppo.update({
        "rollouts":              hp["rollouts"],
        "learning_epochs":       hp["learning_epochs"],
        "mini_batches":          hp["mini_batches"],
        #"discount_factor":       hp["discount_factor"],
        #"lambda":                hp["lambda"],
        #"learning_rate":         hp["learning_rate"],
        #"grad_norm_clip":        hp["grad_norm_clip"],
        #"ratio_clip":            hp["ratio_clip"],
        #"value_clip":            hp["value_clip"],
        #"clip_predicted_values": hp["clip_predicted_values"],
        #"entropy_loss_scale":    hp["entropy_loss_scale"],
        #"value_loss_scale":      hp["value_loss_scale"],
    })

    agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)
    
    trainer = Trainer(cfg=CONFIG["rl"]["trainer"], env=env, agent=agent)
    
    try:
        best_mean_return = - float("inf")
        states, infos = trainer.init_step_train()
        for epoch in range(CONFIG["rl"]["trainer"]["n_epochs"]):
            #############################################################################
            for n in range(cfg_ppo["rollouts"]):
                states, infos, rewards = trainer.step_train(states, infos, n + (epoch * cfg_ppo["rollouts"]))
            #############################################################################
            mean_return = torch.sum(rewards, dim=0).item()
            print(f"Epoch {epoch+1}/{CONFIG['rl']['trainer']['n_epochs']}, mean_return: {mean_return:.3f}")
            if mean_return > best_mean_return:
                best_mean_return = mean_return
            #############################################################################
            trial.report(mean_return, step=epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned() 
            #############################################################################
    finally:
        print("Closing environment and freeing memory...")
        
        env.close() # Force environment close to avoid memory leaks
        print_memory_usage("#### AFTER CLOSE ENV TRIAL END ####")  # Monitor memory after trial
        list_active_tensors("#### AFTER CLOSE ENV TRIAL END ####")

        del env, memory, models, agent, trainer # Delete objects to free memory
        torch.cuda.empty_cache()  # Empty GPU cache
        print_memory_usage("#### AFTER EMPTY CACHE TRIAL END ####")  # Monitor memory after trial
        list_active_tensors("#### AFTER EMPTY CACHE TRIAL END ####")

        gc.collect()  # Manual garbage collection
        print_memory_usage("#### AFTER MANUAL GC TRIAL END ####")  # Monitor memory after trial
        list_active_tensors("#### AFTER MANUAL GC TRIAL END ####")

    return best_mean_return

def main():
    global args
    args = parse_args()

    if CONFIG["set_seed"]:
        set_seed(CONFIG["seed"])
    else:
        CONFIG["seed"] = torch.seed() % (2**32)
        set_seed(CONFIG["seed"])
    
    #################################################################################

    print(CONFIG)

    study = optuna.create_study(
        study_name=f"Satellite_{args.reward_fn}_{datetime.now():%Y%m%d_%H%M%S}",
        storage="sqlite:///optuna_study.db",
        sampler=TPESampler(n_startup_trials=10, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        direction="maximize",
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    ##################################################################

    log_dir = "/home/andreaberti"
    out_path = log_dir + "/optimizer_results/satellite/best_hyperparams.json"
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"\n✅ Salvato in {out_path}")
    print(f"Numero di trials: {len(study.trials)}")
    print(f"➤ mean_return migliore: {study.best_value:.3f}")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    
if __name__ == "__main__":
    main()