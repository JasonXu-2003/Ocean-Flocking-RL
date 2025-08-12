# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu
#!/usr/bin/env python3

"""
experiments.py
统一主控脚本：一键训练、回放、统计与可视化（GIF、曲线、热力图），含多鱼群与障碍透传。

用法示例：
1) 一键全流程（训练→回放→统计/图表）：
   python experiments.py all --algo sac --n_envs 1 --timesteps 200000 \
     --n_whales 3 --obs grid --grid 32 --grid_radius 0.25 \
     --n_schools 2 --fish_per_school 180,120 \
     --obstacles_json '[{"type":"circle","x":0.5,"y":0.5,"r":0.08}]' \
     --disable_builtin_pursuit

2) 只训练：
   python experiments.py rl-train --algo sac --n_envs 4 --timesteps 300000 \
     --n_whales 2 --obs grid --grid 32 --grid_radius 0.25 \
     --n_schools 2 --fish_per_school 150,150 --disable_builtin_pursuit

3) 只回放：
   python experiments.py rl-replay --algo sac --n_whales 2 --obs grid --grid 32 \
     --grid_radius 0.25 --replay_steps 1200 \
     --n_schools 2 --fish_per_school 150,150 --disable_builtin_pursuit

4) 仅统计/图表（用已训练模型评估，输出 CSV+曲线+热力）：
   python experiments.py analyze --algo sac --n_whales 2 --obs grid --grid 32 \
     --grid_radius 0.25 --eval_steps 800 \
     --n_schools 2 --fish_per_school 150,150 --disable_builtin_pursuit

5) 调参（Optuna）：
   python experiments.py tune --trials 30

产物：
- 模型:   assets/models/<algo>_whale.zip
- 回放:   assets/rl_whale_<algo>_w<N>_<obs>[_s<M>].gif
- 统计:   assets/metrics_<algo>_w<N>_<obs>.csv
- 曲线:   assets/capture_curve_<algo>_w<N>_<obs>.png
- 热力:   assets/heatmap_<algo>_w<N>_<obs>.png
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- 基础工具 --------

ASSETS = Path("assets")
MODELS = ASSETS / "models"
ASSETS.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)

def ok(msg): print(f"[OK ] {msg}")
def info(msg): print(f"[INFO] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def err(msg): print(f"[ERR] {msg}")

def run_cmd(cmd_list):
    """子进程运行命令（继承当前python解释器），异常即抛出。"""
    info("RUN: " + " ".join(cmd_list))
    res = subprocess.run(cmd_list, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd_list)}")

def model_path(algo: str) -> Path:
    return MODELS / f"{algo.lower()}_whale.zip"

def gif_path_patterns(algo: str, n_whales: int, obs: str, n_schools: int):
    """优先匹配带 s{n_schools} 的新命名，再退回旧命名。"""
    return [
        ASSETS / f"rl_whale_{algo.lower()}_w{n_whales}_{obs}_s{n_schools}.gif",
        ASSETS / f"rl_whale_{algo.lower()}_w{n_whales}_{obs}.gif",
    ]

def metrics_csv_path(algo: str, n_whales: int, obs: str) -> Path:
    return ASSETS / f"metrics_{algo.lower()}_w{n_whales}_{obs}.csv"

def curve_png_path(algo: str, n_whales: int, obs: str) -> Path:
    return ASSETS / f"capture_curve_{algo.lower()}_w{n_whales}_{obs}.png"

def heatmap_png_path(algo: str, n_whales: int, obs: str) -> Path:
    return ASSETS / f"heatmap_{algo.lower()}_w{n_whales}_{obs}.png"

# -------- 训练 / 回放（调用 train_whales_rl.py） --------

def _append_multi_params(cmd, args):
    """把多鱼群/障碍/控制/设备/并行实现等公共参数透传到 train_whales_rl.py。"""
    cmd += [
        "--n_schools", str(args.n_schools),
    ]
    if args.disable_builtin_pursuit:
        cmd += ["--disable_builtin_pursuit"]
    if args.fish_per_school is not None:
        cmd += ["--fish_per_school", str(args.fish_per_school)]
    if args.obstacles_json is not None:
        cmd += ["--obstacles_json", str(args.obstacles_json)]
    # 新增：设备与向量环境实现
    if getattr(args, "device", None):
        cmd += ["--device", str(args.device)]
    if getattr(args, "vec", None):
        cmd += ["--vec", str(args.vec)]
    return cmd

def do_train(args):
    """调用 train_whales_rl.py 完成训练并保存模型。"""
    py = sys.executable
    cmd = [
        py, "train_whales_rl.py",
        "--mode", "train",
        "--algo", args.algo,
        "--n_envs", str(args.n_envs),
        "--timesteps", str(args.timesteps),
        "--n_fish", str(args.n_fish),
        "--steps_limit", str(args.steps_limit),
        "--n_whales", str(args.n_whales),
        "--obs", args.obs,
        "--grid", str(args.grid),
        "--grid_radius", str(args.grid_radius),
        "--k", str(args.k),
        "--amax", str(args.amax),
        "--seed", str(args.seed),
    ]
    if args.no_flow:
        cmd += ["--no_flow"]
    cmd = _append_multi_params(cmd, args)
    run_cmd(cmd)

    mp = model_path(args.algo)
    if not mp.exists():
        err(f"训练结束但未发现模型文件: {mp}")
        err("请重跑训练或检查 train_whales_rl.py 的保存逻辑。")
        sys.exit(1)
    ok(f"模型已生成 -> {mp}")

def do_replay(args):
    """调用 train_whales_rl.py 完成回放并生成 GIF。"""
    py = sys.executable
    cmd = [
        py, "train_whales_rl.py",
        "--mode", "replay",
        "--algo", args.algo,
        "--n_fish", str(args.n_fish),
        "--steps_limit", str(args.steps_limit),
        "--n_whales", str(args.n_whales),
        "--obs", args.obs,
        "--grid", str(args.grid),
        "--grid_radius", str(args.grid_radius),
        "--k", str(args.k),
        "--amax", str(args.amax),
        "--replay_steps", str(args.replay_steps),
    ]
    if args.no_flow:
        cmd += ["--no_flow"]
    cmd = _append_multi_params(cmd, args)
    run_cmd(cmd)

    # 名称兼容：优先新名（带 s{n_schools}），否则旧名或兜底找到最新
    for p in gif_path_patterns(args.algo, args.n_whales, args.obs, args.n_schools):
        if p.exists():
            ok(f"回放 GIF 已生成 -> {p}")
            return
    warn("未检测到期望命名的 GIF。尝试兜底匹配最新的 rl_whale_*.gif ...")
    gfs = sorted(ASSETS.glob("rl_whale_*.gif"), key=lambda q: q.stat().st_mtime, reverse=True)
    if gfs:
        ok(f"找到最新 GIF -> {gfs[0]}")
    else:
        err("回放后仍未发现任何 GIF。请检查 train_whales_rl.py 的保存逻辑。")
        sys.exit(1)

# -------- 指标统计 / 热力图（直接在此评估模型） --------

def do_analyze(args):
    """
    载入模型，在 env 上执行 eval_steps 步，统计：
    - 每步捕获（按总鱼数减少推得，适配多鱼群）
    - 累计捕获曲线（导出 PNG）
    - 空间热力（将鱼位置计数入栅格）
    - 导出 CSV（step, fish_left, caught_this_step, caught_cum）
    """
    from stable_baselines3 import SAC, PPO
    from ocean.env_ocean import WhaleHuntEnv

    mp = model_path(args.algo)
    if not mp.exists():
        err(f"未找到模型: {mp}\n请先运行训练：python experiments.py rl-train ...")
        sys.exit(1)

    # 解析多鱼群与障碍
    fish_per_school = None
    if args.fish_per_school:
        s = str(args.fish_per_school).strip()
        if s.startswith("["):
            fish_per_school = json.loads(s)
        elif "," in s:
            fish_per_school = [int(x) for x in s.split(",")]
        else:
            fish_per_school = [int(s)]

    obstacles = []
    if args.obstacles_json:
        obstacles = json.loads(str(args.obstacles_json))

    # 创建评估环境（多鱼群/障碍透传）
    env = WhaleHuntEnv(
        n_fish=args.n_fish,
        n_whales=args.n_whales,
        steps_limit=args.steps_limit,
        k_nearest=args.k,
        a_max=args.amax,
        seed=args.seed,
        with_flow=(not args.no_flow),
        obs_mode=args.obs,
        grid_size=args.grid,
        grid_radius=args.grid_radius,
        disable_builtin_pursuit=args.disable_builtin_pursuit,
        n_schools=args.n_schools,
        fish_per_school=fish_per_school,  # ✅ 传 list
        obstacles=obstacles  # ✅ 传 list[dict]
    )

    algo = args.algo.lower()
    model = SAC.load(str(mp)) if algo == "sac" else PPO.load(str(mp))

    obs, _ = env.reset()

    def all_fish_P(env):
        if hasattr(env, "fish_schools"):
            Ps = [sch.P for sch in env.fish_schools if sch.P is not None and sch.P.size > 0]
            return np.vstack(Ps) if Ps else np.zeros((0, 2), dtype=float)
        # 兼容旧单鱼群
        return env.fish.P if hasattr(env, "fish") else np.zeros((0, 2), dtype=float)

    fish_counts, step_catches, cum_catches = [], [], []
    G = args.heatmap_grid
    H = np.zeros((G, G), dtype=float)

    def fish_to_heat(P):
        if P.shape[0] == 0:
            return
        xs = np.linspace(0, env.W.width, G + 1)
        ys = np.linspace(0, env.W.height, G + 1)
        ix = np.clip(np.digitize(P[:, 0], xs) - 1, 0, G - 1)
        iy = np.clip(np.digitize(P[:, 1], ys) - 1, 0, G - 1)
        np.add.at(H, (ix, iy), 1.0)

    P0 = all_fish_P(env)
    prev_n = P0.shape[0]
    total_caught = 0
    fish_to_heat(P0)

    for t in range(args.eval_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(act)

        P = all_fish_P(env)
        cur_n = P.shape[0]
        caught = max(0, prev_n - cur_n)
        total_caught += caught
        prev_n = cur_n

        fish_counts.append(cur_n)
        step_catches.append(caught)
        cum_catches.append(total_caught)

        fish_to_heat(P)

        if done:
            break

    # 导出 CSV
    csv_path = metrics_csv_path(args.algo, args.n_whales, args.obs)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,fish_left,caught_this_step,caught_cum\n")
        for i in range(len(fish_counts)):
            f.write(f"{i},{fish_counts[i]},{step_catches[i]},{cum_catches[i]}\n")
    ok(f"统计CSV -> {csv_path}")

    # 折线图（累计捕获）
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(cum_catches)), cum_catches, lw=2)
    plt.xlabel("Steps"); plt.ylabel("Cumulative Catches")
    plt.title(f"Catches ({args.algo.upper()}, whales={args.n_whales}, obs={args.obs})")
    out_curve = curve_png_path(args.algo, args.n_whales, args.obs)
    fig.tight_layout(); fig.savefig(out_curve, dpi=160); plt.close(fig)
    ok(f"捕获曲线 -> {out_curve}")

    # 热力图（归一化）
    Hn = H / H.max() if H.max() > 0 else H
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(Hn.T, origin="lower", interpolation="nearest")
    plt.colorbar(label="normalized density")
    plt.title("Fish density heatmap (evaluation)")
    out_heat = heatmap_png_path(args.algo, args.n_whales, args.obs)
    fig.tight_layout(); fig.savefig(out_heat, dpi=160); plt.close(fig)
    ok(f"热力图 -> {out_heat}")

# -------- 调参（封装 tune_optuna.py） --------

def do_tune(args):
    py = sys.executable
    cmd = [py, "tune_optuna.py"]
    if args.trials is not None:
        cmd += ["--trials", str(args.trials)]
    run_cmd(cmd)
    # 常见输出文件
    csv = ASSETS / "tune_results.csv"
    png = ASSETS / "tune_scatter.png"
    if csv.exists(): ok(f"Optuna 结果 -> {csv}")
    if png.exists(): ok(f"Optuna 图 -> {png}")

# -------- 主流程（all） --------

def do_all(args):
    info("==> [1/3] 训练")
    do_train(args)
    info("==> [2/3] 回放")
    do_replay(args)
    info("==> [3/3] 统计/图表")
    do_analyze(args)
    ok("全部完成。产物请查看 assets/ 目录。")

# -------- 参数解析 --------

def _parse_fish_per_school(s):
    if s is None: return None
    s = str(s).strip()
    if not s: return None
    if s.startswith("["):
        return s  # 保持 JSON 形式传给 train_whales_rl.py
    if "," in s:
        return s  # 逗号分隔
    # 单数字
    return s

def _parse_obstacles_json(s):
    if s is None: return None
    s = str(s).strip()
    return s if s else None

def build_parser():
    ap = argparse.ArgumentParser(description="Ocean Flocking RL: 实验主控")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 通用默认参数
    def add_common(p):
        p.add_argument("--algo", default="sac", choices=["sac", "ppo"], help="RL 算法")
        p.add_argument("--n_envs", type=int, default=1, help="并行环境数（训练用）")
        p.add_argument("--timesteps", type=int, default=200000, help="训练步数")
        p.add_argument("--n_fish", type=int, default=300)
        p.add_argument("--steps_limit", type=int, default=800)
        p.add_argument("--n_whales", type=int, default=1)
        p.add_argument("--obs", default="grid", choices=["knn", "grid"])
        p.add_argument("--grid", type=int, default=32)
        p.add_argument("--grid_radius", type=float, default=0.25)
        p.add_argument("--k", type=int, default=8, help="knn 模式下最近邻 K")
        p.add_argument("--amax", type=float, default=0.002)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--no_flow", action="store_true", help="关闭流场")
        p.add_argument("--disable_builtin_pursuit", action="store_true",
                       help="True 则只用 RL 控制鲸（默认推荐）")
        # 多鱼群 & 障碍
        p.add_argument("--n_schools", type=int, default=1)
        p.add_argument("--fish_per_school", type=_parse_fish_per_school, default=None,
                       help='如 "150,150" 或 JSON 列表，如 "[120,180]"')
        p.add_argument("--obstacles_json", type=_parse_obstacles_json, default=None,
                       help='JSON 列表，如 \'[{"type":"circle","x":0.5,"y":0.5,"r":0.08}]\'')

        # ★ 新增：设备与并行后端
        p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                       help="训练/回放设备；不传则由 train_whales_rl.py 自动选择")
        p.add_argument("--vec", type=str, default="subproc", choices=["subproc", "dummy"],
                       help="并行环境实现。mac 上如遇多进程卡住可用 dummy 兜底")

    # all
    p_all = sub.add_parser("all", help="一键：训练+回放+统计")
    add_common(p_all)
    p_all.add_argument("--replay_steps", type=int, default=1000)
    p_all.add_argument("--eval_steps", type=int, default=800)
    p_all.add_argument("--heatmap_grid", type=int, default=64)

    # rl-train
    p_train = sub.add_parser("rl-train", help="仅训练")
    add_common(p_train)

    # rl-replay
    p_replay = sub.add_parser("rl-replay", help="仅回放成 GIF")
    add_common(p_replay)
    p_replay.add_argument("--replay_steps", type=int, default=1000)

    # analyze
    p_eval = sub.add_parser("analyze", help="仅统计/图表（用已训练模型）")
    add_common(p_eval)
    p_eval.add_argument("--eval_steps", type=int, default=800)
    p_eval.add_argument("--heatmap_grid", type=int, default=64)

    # tune
    p_tune = sub.add_parser("tune", help="Optuna 自动调参")
    p_tune.add_argument("--trials", type=int, default=None)

    return ap

# -------- 入口 --------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "rl-train":
        do_train(args)
    elif args.cmd == "rl-replay":
        do_replay(args)
    elif args.cmd == "analyze":
        do_analyze(args)
    elif args.cmd == "tune":
        do_tune(args)
    elif args.cmd == "all":
        do_all(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()