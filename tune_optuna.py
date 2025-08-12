# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu
#!/usr/bin/env python3

import os, csv, time
import numpy as np
import optuna
import matplotlib.pyplot as plt

from ocean.core import World, BoidParams, PredatorParams, BoidSchool, Predator

ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

def simulate(n_fish, n_whales, steps, seed, boid_kwargs, pred_kwargs, with_flow=True):
    # 轻量级 headless 仿真（不保存动画）
    def default_flow(x, y, t):
        cx, cy = 0.6, 0.4
        dx, dy = x - cx, y - cy
        w = 0.4 + 0.2*np.sin(0.002*t)
        return (-w*dy, w*dx)

    W = World(flow=default_flow if with_flow else None)
    fish = BoidSchool(n=n_fish, world=W, prm=BoidParams(**boid_kwargs), seed=seed)
    fish.set_goals([(0.95, 0.65, 0.02), (0.2, 0.15, 0.015)])
    whales = [Predator(world=W, prm=PredatorParams(**pred_kwargs), seed=seed+i+1) for i in range(n_whales)]

    eaten_total = 0
    for _ in range(steps):
        for w in whales:
            w.step(fish.P)
        preds = np.vstack([w.P for w in whales])
        fish.step(predators=preds)
        # 捕获
        removed = []
        for w in whales:
            idx = w.capture_indices(fish.P)
            if idx:
                removed.extend(idx)
        if removed:
            removed = np.unique(np.array(removed, dtype=int))
            fish.P = np.delete(fish.P, removed, axis=0)
            fish.V = np.delete(fish.V, removed, axis=0)
            if hasattr(fish, "labels") and fish.labels.shape[0] == eaten_total + n_fish - len(fish.P) - len(removed):
                pass
            if hasattr(fish, "labels") and fish.labels.shape[0] > 0:
                fish.labels = np.delete(fish.labels, removed, axis=0)
            eaten_total += len(removed)

    alive = fish.P.shape[0]
    capture_rate = eaten_total / max(1, steps)        # 越大越好
    # 惩罚：吃太光或吃太少都不好
    penalty = 0.0
    if alive < 0.2 * n_fish:  # 吃得太狠
        penalty -= 0.2
    if eaten_total < 0.05 * n_fish:  # 几乎没捕获
        penalty -= 0.1

    score = capture_rate + penalty
    return dict(score=score, capture_rate=capture_rate, alive=alive, eaten=eaten_total)

def objective(trial: optuna.Trial):
    # 搜索 Boids 与 Predator 的关键参数
    p = dict(
        w_sep = trial.suggest_float("w_sep", 0.8, 2.5),      # 权重：分离（避免拥挤）
        w_ali = trial.suggest_float("w_ali", 0.02, 0.2),     # 权重：对齐（与邻居一致）
        w_coh = trial.suggest_float("w_coh", 0.02, 0.2),     # 权重：聚合（靠近群体中心）
        r_sep = trial.suggest_float("r_sep", 0.01, 0.05),    # 距离：分离作用范围
        r_ali = trial.suggest_float("r_ali", 0.04, 0.12),    # 距离：对齐作用范围
        r_coh = trial.suggest_float("r_coh", 0.06, 0.16),    # 距离：聚合作用范围
    )
    pred = dict(
        vmax   = trial.suggest_float("p_vmax", 0.018, 0.035),
        vmin   = trial.suggest_float("p_vmin", 0.004, 0.010),
        accel  = trial.suggest_float("p_accel", 0.0004, 0.002),
        sense  = trial.suggest_float("p_sense", 0.18, 0.35),
        capture= trial.suggest_float("p_capture", 0.015, 0.035),
    )

    res = simulate(
        n_fish=400, n_whales=2, steps=800, seed=trial.number+42,
        boid_kwargs=p, pred_kwargs=pred, with_flow=True
    )
    # 记录到 trial（便于后处理）
    trial.set_user_attr("alive", res["alive"])
    trial.set_user_attr("eaten", res["eaten"])
    trial.set_user_attr("capture_rate", res["capture_rate"])
    return res["score"]

def main():
    study = optuna.create_study(direction="maximize", study_name="ocean_whales_boids")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # 导出结果
    out_csv = os.path.join(ASSETS, "tune_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "score", "capture_rate", "alive"] + list(study.best_params.keys()))
        for t in study.trials:
            w.writerow([
                t.number, t.value, t.user_attrs.get("capture_rate"), t.user_attrs.get("alive")
            ] + [t.params.get(k) for k in study.best_params.keys()])
    print(f"[OK] CSV -> {out_csv}")

    # 简单散点图（捕获率 vs 终局存活数）
    xs = [t.user_attrs.get("capture_rate", 0.0) or 0.0 for t in study.trials]
    ys = [t.user_attrs.get("alive", 0) or 0 for t in study.trials]
    plt.figure(figsize=(6,4))
    plt.scatter(xs, ys, alpha=0.7)
    plt.xlabel("capture_rate (eaten/steps)")
    plt.ylabel("alive (final)")
    plt.title("Optuna trials: capture vs. alive")
    out_png = os.path.join(ASSETS, "tune_scatter.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[OK] Plot -> {out_png}")

    print("\n[Best]")
    print("score :", study.best_value)
    print("params:", study.best_params)

if __name__ == "__main__":
    # 依赖：pip install optuna
    main()