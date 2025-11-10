import torch
import matplotlib.pyplot as plt
import random
import numpy as np

# === Caricamento del file ===
N_ENV_PLOT = 32  # Numero di ambienti da visualizzare
LOG_PATH = "../Evaluating/logs/trajectories_20251110_091256.pt"
print(f"Caricamento log da: {LOG_PATH}")
data = torch.load(LOG_PATH, map_location="cpu", weights_only=True)

num_records = len(data)
num_envs = data[0]["quat"].shape[0]
print(f"Trovati {num_records} records, {num_envs} ambienti")

# === Stack dei tensori nel tempo ===
steps = [entry["step"] for entry in data]
quat_all   = torch.stack([entry["quat"] for entry in data])      # (T, N, 4)
ang_diff_all = torch.stack([entry["ang_diff"] for entry in data])  # (T, N)
angvel_all = torch.stack([entry["angvel"] for entry in data])    # (T, N, 3)
angacc_all = torch.stack([entry["angacc"] for entry in data])    # (T, N, 3)
actions_all = torch.stack([entry["actions"] for entry in data])  # (T, N, 3)

if N_ENV_PLOT < num_envs:
    env_indices = random.sample(range(num_envs), N_ENV_PLOT)
else:
    env_indices = list(range(num_envs))

def plot_component(title, data_all, labels, non_negative=False, log_scale=False, ylim=None):
    plt.figure(figsize=(14, 8))
    for i, comp in enumerate(labels):
        plt.subplot(len(labels), 1, i + 1)

        # Plot delle traiettorie RANDOM INDICES (di ogni ambiente)
        for env in env_indices:
            plt.plot(steps, data_all[:, env, i], alpha=0.2)
        
        # Calcolo media e deviazione standard
        mean = data_all[:, :, i].mean(dim=1)
        std = data_all[:, :, i].std(dim=1)
        lower = mean - std
        upper = mean + std

        if non_negative:
            lower = np.maximum(lower, 0)
        
        # Plot della media e dell'intervallo di deviazione standard
        plt.plot(steps, mean, color="black", label="media")

        if not log_scale:
            plt.fill_between(steps, lower, upper, color="grey", alpha=1.0)

        plt.title(f"{title} - {comp}")
        plt.ylabel(comp)
        plt.grid(True)

    plt.xlabel("Step")
    plt.tight_layout()

    # Salva la versione lineare
    filename_linear = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename_linear, dpi=600)
    print(f"Grafico lineare salvato: {filename_linear}")

    # Se richiesto il log, salva anche la versione logaritmica
    if log_scale:
        for ax in plt.gcf().axes:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(ylim, ax.get_ylim()[0]))
        filename_log = f"{title.replace(' ', '_').lower()}_log.png"
        plt.savefig(filename_log, dpi=600)
        print(f"Grafico log salvato: {filename_log}")

    plt.close()

# === Plot ===
plot_component("Quaternion", quat_all, ["x", "y", "z", "w"])
plot_component("Angular difference (deg)", ang_diff_all.unsqueeze(-1), ["angle (deg)"], non_negative=True, log_scale=True, ylim=1e-2)
plot_component("Angular velocity", angvel_all, ["x", "y", "z"])
plot_component("Angular acceleration", angacc_all, ["x", "y", "z"])
plot_component("Actions", actions_all, ["x", "y", "z"], log_scale=True, ylim = 1e-4)
