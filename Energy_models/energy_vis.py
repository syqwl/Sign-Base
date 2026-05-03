# 把energy map绘制为 heatmap
# Energy-models/energy_vis.py
import matplotlib.pyplot as plt
import numpy as np

def plot_energy_map(energy_map, title="Energy Map"):
    """
    energy_map: (T,) or (B, T)
    """
    if len(energy_map.shape) == 2:
        energy_map = energy_map.mean(dim=0)  # 取平均
    
    plt.figure(figsize=(10, 3))
    plt.plot(energy_map.cpu().numpy())
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Energy Value")
    plt.tight_layout()
    plt.show()