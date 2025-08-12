# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu
#!/usr/bin/env python3

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ocean.core import (
    World, BoidParams, PredatorParams, BoidSchool, Predator, limit_speed,
    CircleObstacle, PolyObstacle,
    obstacle_repulsion, project_out_of_obstacles
)

class WhaleHuntEnv(gym.Env):
    """
    多鲸 vs 多鱼群 Gymnasium 环境（参数共享协作）。

    观测：
      - obs_mode="knn":  拼接 n_whales 个 [wx, wy, wvx, wvy] + 各自最近K条鱼相对位置（K*2）
      - obs_mode="grid": 拼接 n_whales 个 [wx, wy, wvx, wvy] + 围绕（全体鱼）质心的 GxG 密度栅格（[0,1]）

    动作：长度为 2*n_whales 的加速度向量，范围 [-a_max, a_max]

    奖励：
      r = 1.0 * 新捕获数
          + 0.10 * (上一步鲸到鱼群质心的平均距离 - 当前平均距离)
          - 0.01 * sum(||a_i||)
          - 0.001
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self,
                 n_fish=300,
                 n_whales=1,
                 steps_limit=800,
                 k_nearest=8,
                 a_max=0.002,
                 seed=0,
                 with_flow=True,
                 obs_mode="grid",           # "grid" or "knn"
                 grid_size=32,
                 grid_radius=0.25,          # 质心周围栅格半径（世界坐标）
                 disable_builtin_pursuit=True,  # True: 纯RL；False: RL叠加内置追逐
                 # ---- 多鱼群 & 障碍 ----
                 n_schools=1,
                 fish_per_school=None,      # None->平均分配；或 list，总和= n_fish
                 obstacles=None,            # list[dict] or JSON str
                 # ---- 鲸避障参数 ----
                 whale_obstacle_avoid=True,
                 whale_obs_gain=0.6         # 软避障强度系数（加到速度上，dt=1）
                 ):
        super().__init__()
        assert obs_mode in ("grid", "knn"), "obs_mode must be 'grid' or 'knn'"

        self.n_fish0 = int(n_fish)
        self.n_whales = int(n_whales)
        self.steps_limit = int(steps_limit)
        self.k = int(k_nearest)
        self.a_max = float(a_max)
        self.with_flow = bool(with_flow)
        self.obs_mode = obs_mode
        self.grid_size = int(grid_size)
        self.grid_radius = float(grid_radius)
        self.disable_builtin_pursuit = bool(disable_builtin_pursuit)

        # 鲸避障
        self.whale_obstacle_avoid = bool(whale_obstacle_avoid)
        self.whale_obs_gain = float(whale_obs_gain)

        # 多鱼群与障碍配置
        self.n_schools = int(n_schools)
        self.fish_per_school = None if fish_per_school is None else list(fish_per_school)

        # 障碍配置可为 str(JSON) 或 list[dict]
        if obstacles is None:
            self.obstacles_cfg = []
        elif isinstance(obstacles, str):
            self.obstacles_cfg = json.loads(obstacles)
        else:
            self.obstacles_cfg = list(obstacles)

        self.rng = np.random.default_rng(seed or 0)

        # 观测/动作空间
        if self.obs_mode == "knn":
            per_whale = 4 + self.k * 2
            obs_dim = self.n_whales * per_whale
        else:
            grid_dim = self.grid_size * self.grid_size
            obs_dim = self.n_whales * 4 + grid_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.a_max, high=self.a_max,
                                       shape=(2 * self.n_whales,), dtype=np.float32)

        self._build_world(seed or 0)

    # ---------------- world & reset ----------------

    def _default_flow(self, x, y, t):
        # 简单旋涡流场
        cx, cy = 0.6, 0.4
        dx, dy = x - cx, y - cy
        w = 0.4 + 0.2 * np.sin(0.002 * t)
        return (-w * dy, w * dx)

    def _build_world(self, seed):
        # 开启 hard_obstacle，配合 core.project_out_of_obstacles 防止“穿模”
        self.W = World(flow=self._default_flow if self.with_flow else None,
                       hard_obstacle=True)

        # ---- 障碍注入 ----
        # 支持的配置示例：
        # obstacles = [
        #   {"type":"circle","x":0.5,"y":0.5,"r":0.08},
        #   {"type":"poly","pts":[[0.2,0.2],[0.25,0.3],[0.15,0.28]]}
        # ]
        for cfg in self.obstacles_cfg:
            t = cfg.get("type", "circle")
            if t == "circle":
                ob = CircleObstacle(float(cfg["x"]), float(cfg["y"]), float(cfg["r"]))
                self.W.obstacles.append(ob)
            elif t == "poly":
                pts = [(float(x), float(y)) for x, y in cfg["pts"]]
                ob = PolyObstacle(pts)
                self.W.obstacles.append(ob)

        # ---- 多鱼群 ----
        rng = np.random.default_rng(seed or 0)
        if self.fish_per_school is None:
            base = self.n_fish0 // self.n_schools
            remain = self.n_fish0 - base * self.n_schools
            counts = [base + (1 if i < remain else 0) for i in range(self.n_schools)]
        else:
            counts = list(self.fish_per_school)
            assert sum(counts) == self.n_fish0, "sum(fish_per_school) must equal n_fish"

        self.fish_schools = []
        for i, cnt in enumerate(counts):
            sch = BoidSchool(n=cnt, world=self.W, prm=BoidParams(), seed=(seed or 0) + i)
            # 给每个鱼群一个随机目标点（可选）
            gx = 0.15 + 0.7 * rng.random()
            gy = 0.15 + 0.7 * rng.random()
            sch.set_goals([(gx, gy, 0.02)])
            self.fish_schools.append(sch)

        # ---- 多鲸 ----
        self.whales = [
            Predator(world=self.W, prm=PredatorParams(), seed=(seed or 0) + 101 + i)
            for i in range(self.n_whales)
        ]

        self.t = 0
        self.caught = 0
        self.prev_mean_dist = self._mean_dist_to_centroid()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._build_world(seed if seed is not None else 0)
        return self._observe(), {}

    # ---------------- helpers ----------------

    def _all_fish_P(self):
        if not self.fish_schools:
            return np.zeros((0, 2), dtype=float)
        return np.vstack([sch.P for sch in self.fish_schools])

    def _all_fish_V(self):
        if not self.fish_schools:
            return np.zeros((0, 2), dtype=float)
        return np.vstack([sch.V for sch in self.fish_schools])

    # ---------------- step ----------------

    def step(self, action):
        # --- 动作标准化：兼容标量/一维/二维 ---
        def _norm_action(act):
            a = np.asarray(act, dtype=float)
            if a.ndim == 0:
                return np.full((self.n_whales, 2), float(a))
            if a.ndim == 1:
                if a.size == 2 * self.n_whales:
                    return a.reshape(self.n_whales, 2)
                if a.size == 2 and self.n_whales == 1:
                    return a.reshape(1, 2)
                raise ValueError(f"Invalid action size {a.size}, expected 2 or {2 * self.n_whales}.")
            if a.ndim == 2:
                if a.shape == (self.n_whales, 2):
                    return a
                if a.shape == (1, 2) and self.n_whales == 1:
                    return a
                raise ValueError(f"Invalid action shape {a.shape}, expected ({self.n_whales},2).")
            raise ValueError(f"Unsupported action ndim {a.ndim}")

        a = _norm_action(action)
        a = np.clip(a, -self.a_max, self.a_max)

        # ======== 推进鲸（含可选的避障）========
        # 施加 RL 控制加速度
        for i, w in enumerate(self.whales):
            w.V[0] += a[i]

        if self.whale_obstacle_avoid and getattr(self.W, "obstacles", None):
            # 软避障（对鲸）：把障碍排斥向量当作额外加速度
            WP = np.vstack([w.P[0] for w in self.whales])  # (n_whales, 2)
            AV = obstacle_repulsion(WP, self.W.obstacles, self.W.boundary == "wrap", self.W)
            for i, w in enumerate(self.whales):
                w.V[0] += self.whale_obs_gain * AV[i]

        # 纯 RL（无内置追逐） or 叠加内置追逐
        if self.disable_builtin_pursuit:
            for w in self.whales:
                limit_speed(w.V, w.prm.vmin, w.prm.vmax)
                w.P[0] += w.V[0]
                # 边界
                if self.W.boundary == "wrap":
                    w.P[:, 0] = np.mod(w.P[:, 0], self.W.width)
                    w.P[:, 1] = np.mod(w.P[:, 1], self.W.height)
                else:
                    for d, L in ((0, self.W.width), (1, self.W.height)):
                        if w.P[0, d] > L:
                            w.P[0, d] = 2 * L - w.P[0, d]; w.V[0, d] *= -1
                        if w.P[0, d] < 0:
                            w.P[0, d] = -w.P[0, d]; w.V[0, d] *= -1
        else:
            P_all = self._all_fish_P()
            for w in self.whales:
                w.step(P_all)  # 内置追逐包含速度限制和边界处理

        # 硬投影防穿障碍（对鲸）
        if getattr(self.W, "hard_obstacle", False) and getattr(self.W, "obstacles", None):
            WP = np.vstack([w.P[0] for w in self.whales])
            WV = np.vstack([w.V[0] for w in self.whales])
            project_out_of_obstacles(WP, WV, self.W)
            # 写回
            k = 0
            for w in self.whales:
                w.P[0] = WP[k]; w.V[0] = WV[k]; k += 1

        # ======== 鱼响应（多鱼群）========
        preds = np.vstack([w.P for w in self.whales]) if self.whales else np.zeros((0, 2))
        for sch in self.fish_schools:
            sch.step(predators=preds)

        # ======== 捕获（多鱼群逐一删除）========
        captures = 0
        for sch in self.fish_schools:
            if sch.P.shape[0] == 0:
                continue
            removed = []
            for w in self.whales:
                idx = w.capture_indices(sch.P)
                if idx:
                    removed.extend(idx)
            if removed:
                removed = np.unique(np.array(removed, dtype=int))
                sch.P = np.delete(sch.P, removed, axis=0)
                sch.V = np.delete(sch.V, removed, axis=0)
                if hasattr(sch, "labels") and getattr(sch, "labels", None) is not None and sch.labels.size > 0:
                    sch.labels = np.delete(sch.labels, removed, axis=0)
                captures += len(removed)
        self.caught += captures

        # ======== 奖励 ========
        new_mean_dist = self._mean_dist_to_centroid()
        progress = (self.prev_mean_dist - new_mean_dist)
        control_cost = float(np.sum(np.linalg.norm(a, axis=1)))
        reward = 1.0 * captures + 0.10 * progress - 0.01 * control_cost - 0.001
        self.prev_mean_dist = new_mean_dist

        self.t += 1
        total_now = self._all_fish_P().shape[0]
        terminated = (self.t >= self.steps_limit) or (total_now <= max(5, 0.05 * self.n_fish0))
        truncated = False

        info = {
            "t": self.t,
            "fish_left": int(total_now),
            "caught_step": int(captures),
            "caught_cum": int(self.caught),
            "n_whales": int(self.n_whales),
            "n_schools": int(self.n_schools),
        }

        return self._observe(), float(reward), terminated, truncated, info

    # ---------------- obs helpers ----------------

    def _observe(self):
        # 拼接鲸状态
        whales_state = []
        for w in self.whales:
            whales_state.append(np.concatenate([w.P[0], w.V[0]]))
        whales_state = np.concatenate(whales_state) if whales_state else np.zeros(self.n_whales * 4)

        if self.obs_mode == "knn":
            parts = [whales_state]
            P = self._all_fish_P()
            for w in self.whales:
                if P.shape[0] == 0:
                    nn = np.zeros((self.k, 2), dtype=float)
                else:
                    d = P - w.P[0]
                    dist = np.einsum("ij,ij->i", d, d)
                    k = min(self.k, P.shape[0])
                    idx = np.argpartition(dist, k - 1)[:k] if k > 0 else np.array([], dtype=int)
                    nn = np.zeros((self.k, 2), dtype=float)
                    if k > 0:
                        nn[:k] = d[idx]
                parts.append(nn.flatten())
            obs = np.concatenate(parts).astype(np.float32)
        else:
            grid = self._density_grid()  # (G,G) in [0,1]
            obs = np.concatenate([whales_state, grid.flatten()]).astype(np.float32)
        return obs

    def _density_grid(self):
        """围绕（总鱼）质心统计 GxG 密度栅格，最大值归一化到 [0,1]。"""
        P = self._all_fish_P()
        G = self.grid_size
        cx, cy = self._fish_centroid()
        r = self.grid_radius
        H = np.zeros((G, G), dtype=float)
        if P.shape[0] == 0:
            return H

        xs = np.linspace(cx - r, cx + r, G, endpoint=False)
        ys = np.linspace(cy - r, cy + r, G, endpoint=False)
        dx = (P[:, 0] - xs[0]) / (xs[1] - xs[0] + 1e-9)
        dy = (P[:, 1] - ys[0]) / (ys[1] - ys[0] + 1e-9)
        ix = np.floor(dx).astype(int)
        iy = np.floor(dy).astype(int)
        mask = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)
        if np.any(mask):
            np.add.at(H, (ix[mask], iy[mask]), 1.0)
        if H.max() > 0:
            H /= H.max()
        return H

    def _fish_centroid(self):
        P = self._all_fish_P()
        if P.shape[0] == 0:
            return (self.W.width * 0.5, self.W.height * 0.5)
        c = np.mean(P, axis=0)
        return (float(c[0]), float(c[1]))

    def _mean_dist_to_centroid(self):
        cx, cy = self._fish_centroid()
        d = []
        for w in self.whales:
            dx = w.P[0, 0] - cx
            dy = w.P[0, 1] - cy
            d.append(np.hypot(dx, dy))
        return float(np.mean(d)) if d else 0.0

    def render(self):
        pass