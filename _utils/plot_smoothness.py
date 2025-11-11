import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

TIMESTEPS_TO_PLOT = 1000  # Numero di step da plottare

# === Directory contenente i log ===
LOG_DIR = "../Evaluating/logs/"
log_paths = sorted(glob(os.path.join(LOG_DIR, "trajectories_*.pt")))

print(f"Trovati {len(log_paths)} file log:")
for p in log_paths:
    print(" -", os.path.basename(p))


def load_actions(log_path):
    """Carica le azioni da un file .pt"""
    data = torch.load(log_path, map_location="cpu", weights_only=True)
    actions_all = torch.stack([entry["actions"] for entry in data])  # (T, N, 3)
    return actions_all


def compute_smoothness(actions_all):
    """Calcola media e deviazione standard della smoothness nel tempo"""
    # Calcola differenze tra azioni consecutive
    diffs = actions_all[1:] - actions_all[:-1]  # (T-1, N, 3)
    diff_norm = torch.norm(diffs, dim=-1)       # (T-1, N)
    # Calcola media e deviazione standard su N (batch) per ogni timestep T
    mean_t = diff_norm.mean(dim=1)              # (T-1,)
    std_t  = diff_norm.std(dim=1)               # (T-1,)
    return diff_norm, mean_t, std_t


# === Carica tutti i log e calcola smoothness ===
results = []
names = []

for path in log_paths:
    actions = load_actions(path)
    mean_t, std_t = compute_smoothness(actions)
    results.append((mean_t, std_t))
    names.append(os.path.basename(path))
    print(f"{os.path.basename(path)} -> {len(mean_t)} step")


# === Plot: scala lineare ===
plt.figure(figsize=(14, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

for (mean_t, std_t), name, c in zip(results, names, colors):
    steps = np.arange(len(mean_t))
    # Limita ai primi TIMESTEPS_TO_PLOT step
    if TIMESTEPS_TO_PLOT is not None:
        max_steps = min(TIMESTEPS_TO_PLOT, len(mean_t))
        steps = steps[:max_steps]
        mean_t = mean_t[:max_steps]
        std_t = std_t[:max_steps]

    plt.plot(steps, mean_t, label=name, color=c)
    plt.fill_between(steps, mean_t - std_t, mean_t + std_t, color=c, alpha=0.2)

plt.xlabel("Step")
plt.ylabel("‖Δazione‖ (L2)")
plt.title("Andamento smoothness delle azioni (media ± varianza per run)")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("smoothness_all_runs_mean_std_linear.png", dpi=600)
print("\nGrafico salvato: smoothness_all_runs_mean_std_linear.png")
plt.show()


# === Plot: scala logaritmica ===
plt.figure(figsize=(14, 8))
for (mean_t, std_t), name, c in zip(results, names, colors):
    steps = np.arange(len(mean_t))
    # Limita ai primi TIMESTEPS_TO_PLOT step
    if TIMESTEPS_TO_PLOT is not None:
        max_steps = min(TIMESTEPS_TO_PLOT, len(mean_t))
        steps = steps[:max_steps]
        mean_t = mean_t[:max_steps]
        std_t = std_t[:max_steps]

    # Evita log(0): clamp valori minimi
    mean_t_clamped = torch.clamp(mean_t, min=1e-6)
    std_t_clamped = torch.clamp(std_t, min=1e-6)
    plt.plot(steps, mean_t_clamped, label=name, color=c)
    plt.fill_between(steps, mean_t_clamped - std_t_clamped, mean_t_clamped + std_t_clamped, color=c, alpha=0.2)

plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("‖Δazione‖ (L2) [scala log]")
plt.title("Andamento smoothness delle azioni (scala logaritmica)")
plt.legend(fontsize=8)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("smoothness_all_runs_mean_std_log.png", dpi=600)
print("Grafico salvato: smoothness_all_runs_mean_std_log.png")
plt.show()
