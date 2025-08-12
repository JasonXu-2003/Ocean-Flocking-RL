
# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu
#!/usr/bin/env python3

from ..core import World, BoidParams, PredatorParams, BoidSchool, Predator, CircleObstacle, PolyObstacle
from ..viz import animate_whales
from ..utils.stats import RunStats, save_heatmap
import numpy as np

def default_flow(x, y, t):
    # 简单时间变化的旋涡流场
    cx, cy = 0.6, 0.4
    dx, dy = x - cx, y - cy
    w = 0.4 + 0.2*np.sin(0.002*t)
    return (-w*dy, w*dx)

def run(n_fish=400, n_whales=2, steps=1500, save=None, seed=42, with_flow=True):
    W = World(flow=default_flow if with_flow else None)

    # 鱼群与目标点
    fish = BoidSchool(n=n_fish, world=W, prm=BoidParams(), seed=seed)
    fish.set_goals([(0.95, 0.65, 0.02), (0.2, 0.15, 0.015)])

    # 矿礁 / 障碍
    fish.add_circle_obstacles([CircleObstacle(0.4, 0.5, 0.08), CircleObstacle(0.85, 0.25, 0.06)])
    fish.add_poly_obstacles([PolyObstacle([(0.15,0.6),(0.25,0.68),(0.30,0.60),(0.22,0.52)])])

    # 多条鲸
    whales = [Predator(world=W, prm=PredatorParams(), seed=seed+i+1) for i in range(n_whales)]

    # 统计 & 动画
    stats = RunStats()
    animate_whales(fish, whales, steps=steps, interval=20, save=save, stats=stats)

    # 输出统计与热力图
    base = (save.rsplit(".",1)[0] if save else "whale_fleet")
    stats.save_json(base + "_stats.json")
    np.save(base + "_heat.npy", fish.heat)
    save_heatmap(fish.heat, (W.width, W.height), base + "_heat.png")

if __name__ == "__main__":
    run()
