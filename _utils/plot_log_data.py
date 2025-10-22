import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

log_folder = "../Evaluating/logs"
files = glob.glob(os.path.join(log_folder, "*.csv"))

bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-120", "120-140", "140-160", "160-180"]
metrics = ['controlEffort', 'timeToGoal', 'stayedInGoal', 'precisionOnGoal']
color_map = sns.color_palette("tab10", n_colors=len(files))

for metric in metrics:
    plt.figure(figsize=(10,5))

    for i, file in enumerate(files[1:]):  # SKIP FIRST FILE (evaluation_log_0.csv)
        df = pd.read_csv(file)
        
        print(f"\n--- {file} ---")
        print(df.columns.tolist())
        print(df.head())

        df['DistanceRange'] = pd.cut(df['startGoalDistance'], bins=bins, labels=labels, include_lowest=True)
        agg = df.groupby('DistanceRange')[metric].mean().reset_index()

        label = os.path.splitext(os.path.basename(file))[0]

        plt.plot(agg['DistanceRange'], agg[metric], 
                 marker='o', markersize=7, alpha=0.6,
                 color=color_map[i], label=label)

    plt.xlabel('Distance Range from Goal')
    plt.ylabel(f'Average {metric}')
    plt.title(f'{metric} by Distance from Goal', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{metric}.png', dpi=300)
    plt.show()
