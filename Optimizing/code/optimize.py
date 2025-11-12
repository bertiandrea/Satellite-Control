# optimize.py

from code.configs.satellite_config import CONFIG
from code.envs.satellite import Satellite
from code.models.custom_model import Shared
from code.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper

import isaacgym #BugFix
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from code.trainer.trainer import Trainer # Custom Trainer

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

def sample_ppo_params(trial: optuna.Trial):
    return {
        "learning_epochs": trial.suggest_categorical("learning_epochs", [8, 16]),
    }

def print_memory_usage(tag=""):
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_mem = process.memory_info().rss / 1024**2  # in MB
    print(f"\n[MEMORY] {tag}")
    print(f"  CPU RSS: {cpu_mem:.2f} MB")
    mem_alloc = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"  GPU allocated: {mem_alloc:.2f} MB | reserved: {mem_reserved:.2f} MB")
    print("#" * len(f"[MEMORY] {tag}"))

def objective(trial: optuna.Trial) -> float:
    print_memory_usage("#### BEFORE TRIAL START ####")  # Monitor memory before trial

    env = Satellite(
        cfg=CONFIG,
        rl_device=CONFIG["rl_device"],
        sim_device=CONFIG["sim_device"],
        graphics_device_id=CONFIG["graphics_device_id"],
        headless=CONFIG["headless"],
        virtual_screen_capture=CONFIG["virtual_screen_capture"],
        force_render= CONFIG["force_render"],
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=CONFIG["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Shared(env.state_space, env.action_space, env.device)
    models["value"] = models["policy"]  # Shared model for policy and value

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update(CONFIG["rl"]["PPO"])
    cfg_ppo.update(sample_ppo_params(trial))

    agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)
    
    trainer = Trainer(cfg=CONFIG["rl"]["trainer"], env=env, agent=agent)
    
    try:
        best_mean_return = -float("inf")
        states, infos = trainer.init_step_train()
        for epoch in range(CONFIG["rl"]["trainer"]["n_epochs"]):
            for n in range(cfg_ppo["rollouts"]):
                states, infos, rewards = trainer.step_train(states, infos, n + (epoch * cfg_ppo["rollouts"]))
            mean_return = torch.sum(rewards, dim=0).item()
            print(f"Epoch {epoch+1}/{CONFIG['rl']['trainer']['n_epochs']}, mean_return: {mean_return:.3f}")
            best_mean_return = max(best_mean_return, mean_return)
            trial.report(mean_return, step=epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned() 
    finally:
        print("Closing environment and freeing memory...")
        env.close() # Force environment close to avoid memory leaks
        print_memory_usage("#### AFTER CLOSE ENV TRIAL END ####")  # Monitor memory after trial
        del env, memory, models, agent, trainer # Delete objects to free memory
        gc.collect()  # Manual garbage collection
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Empty GPU cache
        print_memory_usage("#### AFTER GC ####")

    return best_mean_return

def main():
    study = optuna.create_study(
        study_name=f"Satellite_{datetime.now():%Y%m%d_%H%M%S}",
        storage="sqlite:///optuna_study.db",
        sampler=TPESampler(n_startup_trials=10, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        direction="maximize",
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, 
                       callbacks=[lambda study, trial: gc.collect()])
    except KeyboardInterrupt:
        pass

    ##################################################################

    log_dir = "/home/andreaberti"
    out_path = log_dir + "/optimizer_results/satellite/best_hyperparams.json"
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