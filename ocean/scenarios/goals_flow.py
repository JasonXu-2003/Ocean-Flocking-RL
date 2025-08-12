# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu

#!/usr/bin/env python3
from ..core import World, BoidParams, BoidSchool, CircleObstacle
from ..viz import animate_goals
from ..utils.stats import RunStats, save_heatmap
import numpy as np

def run(n_fish=300, steps=1200, save=None, seed=0):
    # no global flow, just goals + obstacles
    W = World(flow=None)
    fish = BoidSchool(n=n_fish, world=W, prm=BoidParams(), seed=seed)
    fish.set_goals([(0.2, 0.7, 0.02), (0.9, 0.4, 0.015), (0.6, 0.2, 0.01)])
    fish.add_circle_obstacles([CircleObstacle(0.55, 0.45, 0.07)])
    stats = RunStats()
    animate_goals(fish, steps=steps, interval=20, save=save, stats=stats)

    if save:
        base = save.rsplit(".",1)[0]
    else:
        base = "goals"
    stats.save_json(base + "_stats.json")
    np.save(base + "_heat.npy", fish.heat)
    save_heatmap(fish.heat, (W.width, W.height), base + "_heat.png")

if __name__ == "__main__":
    run()
