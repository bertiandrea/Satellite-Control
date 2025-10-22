import math
import matplotlib.pyplot as plt
import numpy as np

# Parameters
changing_steps = [
            5000, 7500, 10000, 
            12500, 15000, 17500, 20000,
            22500, 25000, 27500, 30000,
            32500, 35000, 37500, 40000,
            42500, 45000, 47500, 50000,
            52500, 55000, 57500, 60000,
        ]
target_deg = 10 * (math.pi / 180)
final_target_deg = 0.1 * (math.pi / 180)
r_at_target = 0.95

# Sigma calculation
sigma_fix_value = target_deg / math.sqrt(-2 * math.log(r_at_target))
n = len(changing_steps)
decay_rate = math.log(final_target_deg / target_deg) / (n - 1)
sigma_exp = [target_deg * math.exp(decay_rate * i) / math.sqrt(-2 * math.log(r_at_target)) for i in range(n)]
sigma_lin = [(target_deg + (final_target_deg - target_deg) * (i / (n-1))) / math.sqrt(-2 * math.log(r_at_target)) for i in range(n)]

# Theta range for reward plot
theta = np.linspace(0, 90 * math.pi / 180, 500)  # 0 to ~5 degrees

plt.figure(figsize=(8,5))

for i, step in enumerate(changing_steps):
    reward_exp = np.exp(-theta**2 / (2 * sigma_exp[i]**2))
    reward_lin = np.exp(-theta**2 / (2 * sigma_lin[i]**2))
    
    plt.plot(theta * 180 / math.pi, reward_exp, color='red', linestyle='-', label=f'Exp decay (step {step})')
    plt.plot(theta * 180 / math.pi, reward_lin, color='blue', linestyle='--', label=f'Lin decay (step {step})')

reward_fix = np.exp(-theta**2 / (2 * sigma_fix_value**2))
plt.plot(theta * 180 / math.pi, reward_fix, color='black', linestyle=':', label=f'Fixed reward')

plt.xlabel('Error (degrees)')
plt.ylabel('Reward')
plt.title('Reward shape')
plt.grid(True)
plt.legend()
plt.savefig('reward_shape.png')
plt.show()
