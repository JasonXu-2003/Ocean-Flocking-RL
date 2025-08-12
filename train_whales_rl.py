# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu
#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import warnings

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# ---- 设备/性能相关：自动选择 MPS/CUDA/CPU，适度提升并行线程 ----
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 不支持时回退CPU

try:
    import torch
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
except Exception:
    torch = None

# 避免训练阶段加载 pyplot（会卡住多进程）
# 回放时我们再按需 import matplotlib

from ocean.env_ocean import WhaleHuntEnv

ASSETS = "assets"
MODEL_DIR = os.path.join(ASSETS, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- utils ----------------

def _auto_device():
    """优先 MPS（Apple GPU） -> CUDA -> CPU"""
    if torch is None:
        return "cpu"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _parse_fish_per_school(s):
    """
    返回 None 或 list[int]。支持:
      - None / "" -> None
      - "150,150" -> [150, 150]
      - "[120,180]" -> [120, 180]
      - "300" -> [300]
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.startswith("["):
        return list(json.loads(s))
    if "," in s:
        return [int(x) for x in s.split(",") if str(x).strip() != ""]
    return [int(s)]


def _parse_obstacles_json(s):
    """
    返回 list[dict]，或 []。
    """
    if not s:
        return []
    s = str(s).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _count_groups(env):
    """估算群数量：按每个 school 的 labels 唯一数；否则退化为学校数量/1。"""
    if hasattr(env, "fish_schools") and env.fish_schools:
        cnt = 0
        for sch in env.fish_schools:
            if hasattr(sch, "labels") and sch.P is not None and sch.P.size > 0:
                cnt += len(np.unique(sch.labels))
        if cnt == 0:
            cnt = len(env.fish_schools)
        return int(cnt)
    return 1


def _total_fish(env):
    if hasattr(env, "fish_schools") and env.fish_schools:
        return int(sum(sch.P.shape[0] for sch in env.fish_schools))
    if hasattr(env, "fish"):
        return int(env.fish.P.shape[0])
    return 0


# ---------------- env factory ----------------

def make_env(seed, n_fish, steps_limit, k, amax, with_flow,
             n_whales, obs_mode, grid_size, grid_radius,
             disable_builtin_pursuit,
             # NEW:
             n_schools, fish_per_school, obstacles):
    def _fn():
        env = WhaleHuntEnv(
            n_fish=n_fish, n_whales=n_whales, steps_limit=steps_limit,
            k_nearest=k, a_max=amax, seed=seed, with_flow=with_flow,
            obs_mode=obs_mode, grid_size=grid_size, grid_radius=grid_radius,
            disable_builtin_pursuit=disable_builtin_pursuit,
            n_schools=n_schools, fish_per_school=fish_per_school,
            obstacles=obstacles
        )
        return env
    return _fn


# ---------------- train ----------------

def train(args):
    print("[BOOT] train_whales_rl.py :: TRAIN mode")
    print(f"[CFG] algo={args.algo} n_envs={args.n_envs} timesteps={args.timesteps}")
    print(f"[CFG] whales={args.n_whales} obs={args.obs} grid={args.grid} r={args.grid_radius} "
          f"k={args.k} amax={args.amax} flow={not args.no_flow}")
    print(f"[CFG] schools={args.n_schools} fish_per_school={args.fish_per_school} "
          f"disable_builtin_pursuit={args.disable_builtin_pursuit}")
    if args.obstacles_json:
        print(f"[CFG] obstacles_json={str(args.obstacles_json)[:120]}...")

    # 设备：命令行未指定则自动选择
    device = (args.device or "").strip().lower() if hasattr(args, "device") else ""
    if device not in ("cpu", "cuda", "mps"):
        device = _auto_device()
    print(f"[DEV] using device: {device}")

    set_random_seed(args.seed)

    fish_per_school = _parse_fish_per_school(args.fish_per_school)
    obstacles = _parse_obstacles_json(args.obstacles_json)

    env_fns = [
        make_env(
            seed=args.seed + i,
            n_fish=args.n_fish,
            steps_limit=args.steps_limit,
            k=args.k,
            amax=args.amax,
            with_flow=not args.no_flow,
            n_whales=args.n_whales,
            obs_mode=args.obs,
            grid_size=args.grid,
            grid_radius=args.grid_radius,
            disable_builtin_pursuit=args.disable_builtin_pursuit,
            n_schools=args.n_schools,
            fish_per_school=fish_per_school,
            obstacles=obstacles
        )
        for i in range(args.n_envs)
    ]

    # ---- VecEnv：默认 subproc；提供 --vec=dummy 兜底 ----
    if args.vec == "dummy" or args.n_envs == 1:
        vec = DummyVecEnv([env_fns[0]])
        print("[OK] VecEnv created (DummyVecEnv).")
    else:
        # 显式使用 spawn，macOS 更稳
        vec = SubprocVecEnv(env_fns, start_method="spawn")
        print("[OK] VecEnv created (SubprocVecEnv, spawn).")

    print("[INFO] observation_space:", vec.observation_space)
    print("[INFO] action_space     :", vec.action_space)

    # smoke test（确保不会静默退出）
    try:
        print("[SMOKE] resetting vec env ...")
        obs = vec.reset()
        print("[SMOKE] step(1) ...")
        a = vec.action_space.sample()
        if args.n_envs == 1 or args.vec == "dummy":
            a = np.expand_dims(a, axis=0)
        obs, rew, done, infos = vec.step(a)
        print("[OK] Smoke test passed (1 step).")
    except Exception as e:
        print("[ERR] Smoke test failed:", repr(e))
        raise

    policy_kwargs = dict(net_arch=[256, 256])
    tb_dir = os.path.join(ASSETS, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy", vec, verbose=1,
            learning_rate=3e-4, n_steps=1024, batch_size=256,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
            policy_kwargs=policy_kwargs, seed=args.seed,
            tensorboard_log=tb_dir,
            device=device
        )
    else:
        model = SAC(
            "MlpPolicy", vec, verbose=1,
            learning_rate=3e-4, batch_size=256,
            gamma=0.99, tau=0.02, train_freq=64, gradient_steps=64,
            policy_kwargs=policy_kwargs, seed=args.seed,
            tensorboard_log=tb_dir,
            device=device
        )

    print("[OK] Model built. Start learning...")
    model.learn(total_timesteps=int(args.timesteps))

    out = os.path.join(MODEL_DIR, f"{args.algo}_whale.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    print(f"[OK] Training done. Model saved -> {out}")
    print(f"[TIP] TensorBoard: tensorboard --logdir {tb_dir}")


# ---------------- replay ----------------

def replay(args):
    print("[BOOT] train_whales_rl.py :: REPLAY mode")

    # 回放也用同一策略；不指定则自动
    device = (args.device or "").strip().lower() if hasattr(args, "device") else ""
    if device not in ("cpu", "cuda", "mps"):
        device = _auto_device()
    print(f"[DEV] using device: {device}")

    # ONLY NOW import matplotlib（训练阶段完全不 import）
    import matplotlib
    matplotlib.use("Agg")  # 强制无界面
    import matplotlib.pyplot as plt

    fish_per_school = _parse_fish_per_school(args.fish_per_school)
    obstacles = _parse_obstacles_json(args.obstacles_json)

    env = WhaleHuntEnv(
        n_fish=args.n_fish, n_whales=args.n_whales, steps_limit=args.steps_limit,
        k_nearest=args.k, a_max=args.amax, with_flow=not args.no_flow,
        obs_mode=args.obs, grid_size=args.grid, grid_radius=args.grid_radius,
        disable_builtin_pursuit=args.disable_builtin_pursuit,
        n_schools=args.n_schools, fish_per_school=fish_per_school,
        obstacles=obstacles
    )

    model_path = os.path.join(MODEL_DIR, f"{args.algo}_whale.zip")
    if not os.path.exists(model_path):
        print(f"[ERR] model not found: {model_path}\n"
              f"请先训练：python train_whales_rl.py --mode train --algo {args.algo}")
        return

    loader = SAC if args.algo == "sac" else PPO
    model = loader.load(model_path, device=device)
    print(f"[OK] loaded model -> {model_path}")

    obs, _ = env.reset()

    # --------- 一次建图（更快）---------
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.set_xlim(0, env.W.width)
    ax.set_ylim(0, env.W.height)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(plt.Rectangle((0, 0), env.W.width, env.W.height, fill=False, linewidth=1.0))

    # 画障碍（字段使用 cx, cy, r）
    if getattr(env.W, "obstacles", None):
        for ob in env.W.obstacles:
            if hasattr(ob, "r") and hasattr(ob, "cx") and hasattr(ob, "cy"):
                circ = plt.Circle((float(ob.cx), float(ob.cy)), float(ob.r),
                                  fill=False, lw=2, ls="--", ec="red")
                ax.add_patch(circ)
            elif hasattr(ob, "pts"):
                pts = np.asarray(ob.pts, dtype=float)
                if pts.ndim == 2 and pts.shape[0] >= 2:
                    ax.plot(np.r_[pts[:, 0], pts[0, 0]],
                            np.r_[pts[:, 1], pts[0, 1]],
                            "r--", lw=2)

    # 为每个鱼群建 scatter 句柄
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    school_scatters = []
    if hasattr(env, "fish_schools") and env.fish_schools:
        for si, _sch in enumerate(env.fish_schools):
            sc = ax.scatter([], [], s=8, c=colors[si % len(colors)], alpha=0.95, label=f"school{si+1}")
            school_scatters.append(sc)
    else:
        sc = ax.scatter([], [], s=8, c=colors[0], alpha=0.95, label="school")
        school_scatters.append(sc)

    # 鲸的 scatter
    whale_sc = ax.scatter([], [], s=160, marker='>', edgecolors='black', facecolors='none', linewidths=1.5, label="whales")

    # HUD 文本（更多关键信息）
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left')

    frames = []
    eaten_cum = 0
    total_fish_prev = _total_fish(env)

    for t in range(args.replay_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)

        # 更新鱼群坐标
        if hasattr(env, "fish_schools") and env.fish_schools:
            for si, sch in enumerate(env.fish_schools):
                if sch.P.shape[0] > 0:
                    school_scatters[si].set_offsets(sch.P)
                else:
                    school_scatters[si].set_offsets(np.zeros((0, 2)))
        else:
            if hasattr(env, "fish") and env.fish.P.shape[0] > 0:
                school_scatters[0].set_offsets(env.fish.P)
            else:
                school_scatters[0].set_offsets(np.zeros((0, 2)))

        # 更新鲸坐标
        WXY = np.vstack([w.P for w in env.whales]) if env.whales else np.zeros((0, 2))
        whale_sc.set_offsets(WXY)

        # 统计 HUD
        total_fish_now = _total_fish(env)
        caught_step = max(0, total_fish_prev - total_fish_now)
        eaten_cum += caught_step
        total_fish_prev = total_fish_now
        groups = _count_groups(env)
        hud.set_text(
            f"t={t}  fish={total_fish_now}  caught_step={caught_step}  "
            f"caught_cum={eaten_cum}  groups≈{groups}"
        )

        # 抓帧
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8) \
            .reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame.copy())

        if done:
            break

    if len(frames) == 0:
        print("[ERR] no frames captured; replay aborted")
        return

    try:
        import imageio
    except ImportError:
        import sys as _sys
        os.system(f"{_sys.executable} -m pip install imageio")
        import imageio

    os.makedirs(ASSETS, exist_ok=True)
    tag = f"w{args.n_whales}_{args.obs}_s{args.n_schools}"
    out_gif = os.path.join(ASSETS, f"rl_whale_{args.algo}_{tag}.gif")
    imageio.mimsave(out_gif, frames, duration=1/20.0)
    print(f"[OK] replay gif -> {out_gif} (frames={len(frames)})")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="RL training/replay for Whale Hunt")
    ap.add_argument("--mode", choices=["train", "replay"], required=True)
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=200000)

    ap.add_argument("--n_fish", type=int, default=300)
    ap.add_argument("--steps_limit", type=int, default=800)
    ap.add_argument("--n_whales", type=int, default=1)

    ap.add_argument("--obs", choices=["knn", "grid"], default="grid")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--grid_radius", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=8)

    ap.add_argument("--amax", type=float, default=0.002)
    ap.add_argument("--no_flow", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_builtin_pursuit", action="store_true",
                    help="True 则只用 RL 控制鲸（默认推荐）")

    # NEW: multi-school & obstacles
    ap.add_argument("--n_schools", type=int, default=1)
    ap.add_argument("--fish_per_school", type=str, default=None,
                    help='如 "150,150" 或 JSON 列表，如 "[120,180]"')
    ap.add_argument("--obstacles_json", type=str, default=None,
                    help='JSON 列表，如 \'[{"type":"circle","x":0.5,"y":0.5,"r":0.08}]\'')
    ap.add_argument("--replay_steps", type=int, default=1000)

    # 可选：手动指定设备（cpu / cuda / mps），不传则自动
    ap.add_argument("--device", type=str, default=None)

    # 可选：选择 VecEnv 实现（subproc/dummy）
    ap.add_argument("--vec", type=str, default="subproc", choices=["subproc", "dummy"])

    args = ap.parse_args()

    if args.mode == "train":
        return train(args)
    else:
        return replay(args)


if __name__ == "__main__":
    main()