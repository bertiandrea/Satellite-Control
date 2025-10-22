import torch
import matplotlib.pyplot as plt
import random

# === Caricamento del file ===
N_ENV_PLOT = 100  # Numero di ambienti da visualizzare
LOG_PATH = "./trajectories.pt"
print(f"Caricamento log da: {LOG_PATH}")
data = torch.load(LOG_PATH, map_location="cpu", weights_only=True)

num_records = len(data)
num_envs = data[0]["quat"].shape[0]
print(f"Trovati {num_records} records, {num_envs} ambienti")

# === Stack dei tensori nel tempo ===
steps = [entry["step"] for entry in data]
quat_all   = torch.stack([entry["quat"] for entry in data])      # (T, N, 4)
angvel_all = torch.stack([entry["angvel"] for entry in data])    # (T, N, 3)
angacc_all = torch.stack([entry["angacc"] for entry in data])    # (T, N, 3)
actions_all = torch.stack([entry["actions"] for entry in data])  # (T, N, 3)

if N_ENV_PLOT < num_envs:
    env_indices = random.sample(range(num_envs), N_ENV_PLOT)
else:
    env_indices = list(range(num_envs))

def plot_component(title, data_all, labels):
    plt.figure(figsize=(10, 6))
    for i, comp in enumerate(labels):
        plt.subplot(len(labels), 1, i + 1)

        # Plot delle traiettorie RANDOM INDICES (di ogni ambiente)
        for env in env_indices:
            plt.plot(steps, data_all[:, env, i], alpha=0.2)
        
        # Calcolo media e deviazione standard
        mean = data_all[:, :, i].mean(dim=1)
        std = data_all[:, :, i].std(dim=1)
        # Plot della media e dell'intervallo di deviazione standard
        plt.plot(steps, mean, color="black", label="media")
        plt.fill_between(steps, mean - std, mean + std, color="grey", alpha=1.0)

        plt.title(f"{title} - {comp}")
        plt.ylabel(comp)
        plt.grid(True)
    plt.xlabel("Step")
    plt.tight_layout()

    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Grafico salvato: {filename}")

# === Plot ===
plot_component("Quaternion", quat_all, ["x", "y", "z", "w"])
plot_component("Angular velocity", angvel_all, ["x", "y", "z"])
plot_component("Angular acceleration", angacc_all, ["x", "y", "z"])
plot_component("Actions", actions_all, ["x", "y", "z"])

