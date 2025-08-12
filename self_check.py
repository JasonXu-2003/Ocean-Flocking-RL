# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu
#!/usr/bin/env python3

import os, sys, argparse, traceback, numpy as np

def ok(msg): print("[OK ]", msg)
def err(msg): print("[ERR]", msg)

def check_env_basic():
    from ocean.env_ocean import WhaleHuntEnv
    env = WhaleHuntEnv(n_fish=200, n_whales=1, steps_limit=200, obs_mode="knn", k_nearest=6, a_max=0.002, with_flow=True)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape, f"obs shape {obs.shape} != {env.observation_space.shape}"
    a = env.action_space.sample()
    # 支持 (2,) 或 (1,2)
    env.step(a)
    ok("Env reset/step (knn)")

    env = WhaleHuntEnv(n_fish=200, n_whales=2, steps_limit=200, obs_mode="grid", grid_size=16, grid_radius=0.25)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    a = env.action_space.sample()
    env.step(a)
    ok("Env reset/step (grid, multi-whale)")

def check_vecenv_and_smoke():
    from ocean.env_ocean import WhaleHuntEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    def mf(seed):
        def _f():
            return WhaleHuntEnv(n_fish=150, n_whales=1, steps_limit=100, obs_mode="knn", k_nearest=4, a_max=0.002, seed=seed)
        return _f

    v1 = DummyVecEnv([mf(0)])
    a = v1.action_space.sample()
    a = np.expand_dims(a, 0)
    v1.step(a); ok("DummyVecEnv smoke")

    v2 = SubprocVecEnv([mf(1), mf(2)])
    a = np.stack([v2.action_space.sample() for _ in range(v2.num_envs)], axis=0)  # (n_envs, action_dim)
    v2.step(a)

def quick_train_and_replay():
    import matplotlib
    matplotlib.use("Agg")  # 后端防止无显示出错
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ocean.env_ocean import WhaleHuntEnv
    import imageio

    # 训练少量步数
    def mf(seed):
        def _f():
            return WhaleHuntEnv(n_fish=200, n_whales=1, steps_limit=300, obs_mode="knn", k_nearest=6, a_max=0.002, seed=seed)
        return _f
    vec = DummyVecEnv([mf(0)])
    model = SAC("MlpPolicy", vec, verbose=0, learning_rate=3e-4, batch_size=128, gamma=0.99, tau=0.02,
                train_freq=32, gradient_steps=32, policy_kwargs=dict(net_arch=[128,128]))
    model.learn(total_timesteps=1500)
    os.makedirs("assets/models", exist_ok=True)
    model_path = "assets/models/_smoke_sac_whale.zip"
    model.save(model_path)
    assert os.path.exists(model_path)
    ok(f"Model saved -> {model_path}")

    # 回放并生成 gif
    env = WhaleHuntEnv(n_fish=200, n_whales=1, steps_limit=200, obs_mode="knn", k_nearest=6, a_max=0.002)
    mdl = SAC.load(model_path)
    obs, _ = env.reset()
    frames = []
    import matplotlib.pyplot as plt
    for t in range(120):
        act, _ = mdl.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(act)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.set_xlim(0, env.W.width); ax.set_ylim(0, env.W.height)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal', adjustable='box')
        if env.fish.P.shape[0] > 0:
            ax.scatter(env.fish.P[:,0], env.fish.P[:,1], s=6)
        import numpy as np
        WXY = np.vstack([w.P for w in env.whales])
        ax.scatter(WXY[:,0], WXY[:,1], s=120, marker='>', edgecolors='black', facecolors='none', linewidths=1.2)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame); plt.close(fig)
        if done: break
    os.makedirs("assets", exist_ok=True)
    out_gif = "assets/_smoke_rl.gif"
    imageio.mimsave(out_gif, frames, duration=1/20.0)
    assert os.path.exists(out_gif)
    ok(f"Replay GIF saved -> {out_gif}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="更快更短的检查（默认已很快）")
    args = ap.parse_args()
    try:
        check_env_basic()
        check_vecenv_and_smoke()
        quick_train_and_replay()
        print("\n[ALL GREEN] self_check passed.")
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        err("self_check failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
