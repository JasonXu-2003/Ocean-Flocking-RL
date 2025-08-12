# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu

#!/usr/bin/env python3
"""
Stats logger and heatmap utilities for experiments.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

class RunStats:
    def __init__(self):
        self.t = []
        self.n_fish = []
        self.eaten_total = []
        self.mean_speed = []

    def log(self, t, n_fish, eaten_total, V):
        self.t.append(float(t))
        self.n_fish.append(int(n_fish))
        self.eaten_total.append(int(eaten_total))
        self.mean_speed.append(float(np.linalg.norm(V, axis=1).mean()))

    def save_json(self, path):
        data = dict(t=self.t, n_fish=self.n_fish, eaten_total=self.eaten_total, mean_speed=self.mean_speed)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def save_heatmap(heat, world_size, out_png, cmap=None):
    H = heat.T  # transpose for intuitive x-y orientation
    H = H / (H.max() + 1e-9)
    plt.figure(figsize=(6,4))
    plt.imshow(H, origin='lower', extent=[0, world_size[0], 0, world_size[1]], aspect='auto', cmap=cmap or 'viridis')
    plt.colorbar(label="normalized occupancy")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
