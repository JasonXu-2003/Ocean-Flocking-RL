# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu

#!/usr/bin/env python3
"""
Core entities:
- World: size, boundary, flow field, obstacles (+ hard projection)
- Obstacles: circles & polygons
- SpatialGrid: neighbor queries
- BoidSchool: fish school with split/merge behavior, goals, predator avoidance, obstacle avoidance
- Predator: whale (pursuit), supports multiple instances
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Union
import numpy as np

Vec = np.ndarray

# ======================= World, Obstacles, Flow =======================

@dataclass
class World:
    width: float = 1.2
    height: float = 0.8
    boundary: str = "wrap"   # "wrap" or "reflect"
    cell: float = 0.06
    # flow field u(x,y,t) -> (ux, uy); can be None
    flow: Optional[Callable[[float, float, float], Tuple[float, float]]] = None
    # global obstacles (circles & polygons)
    obstacles: List["Obstacle"] = field(default_factory=list)
    # 硬投影开关（True: 穿越时强制推出障碍外，配合软避障更稳定）
    hard_obstacle: bool = True

    # 便捷方法
    def add_obstacle(self, ob: "Obstacle"):
        self.obstacles.append(ob)

    def add_obstacles(self, obs: List["Obstacle"]):
        self.obstacles.extend(obs)


@dataclass
class CircleObstacle:
    cx: float
    cy: float
    r: float

    def vector_away(self, p: Vec, wrap: bool, W: World) -> Vec:
        dx = p[0] - self.cx
        dy = p[1] - self.cy
        if wrap:
            if dx > 0.5 * W.width:  dx -= W.width
            if dx < -0.5 * W.width: dx += W.width
            if dy > 0.5 * W.height: dy -= W.height
            if dy < -0.5 * W.height: dy += W.height
        d2 = dx*dx + dy*dy
        if d2 <= self.r*self.r:
            # inside: push strong
            return np.array([dx, dy]) / (d2 + 1e-9) * 0.08
        elif d2 < (1.8*self.r)*(1.8*self.r):
            # near boundary: gentle push
            return np.array([dx, dy]) / (d2 + 1e-9) * 0.03
        else:
            return np.zeros(2)


@dataclass
class PolyObstacle:
    pts: List[Tuple[float, float]]  # simple polygon (convex preferred)

    # 点是否在多边形内（ray casting）
    def _inside(self, x: float, y: float) -> bool:
        c = False
        n = len(self.pts)
        for i in range(n):
            x1, y1 = self.pts[i]
            x2, y2 = self.pts[(i+1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1 + 1e-12) + x1):
                c = not c
        return c

    def _centroid(self) -> np.ndarray:
        cx = sum(x for x, _ in self.pts) / len(self.pts)
        cy = sum(y for _, y in self.pts) / len(self.pts)
        return np.array([cx, cy], dtype=float)

    def vector_away(self, p: Vec, wrap: bool, W: World) -> Vec:
        # outward relative to centroid (simple & stable)
        c = self._centroid()
        d = p - c
        inside = self._inside(float(p[0]), float(p[1]))
        if inside:
            return d / (np.linalg.norm(d) + 1e-9) * 0.08
        # near edge -> mild push
        mind = 1e9
        for i in range(len(self.pts)):
            a = np.array(self.pts[i], dtype=float)
            b = np.array(self.pts[(i+1) % len(self.pts)], dtype=float)
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
            proj = a + t * ab
            dist = np.linalg.norm(p - proj)
            if dist < mind: mind = dist
        if mind < 0.06:
            return d / (np.linalg.norm(d) + 1e-9) * 0.03
        return np.zeros(2)


Obstacle = Union[CircleObstacle, PolyObstacle]


def obstacle_repulsion(P: np.ndarray, obstacles: List[Obstacle], wrap: bool, W: World) -> np.ndarray:
    """
    软避障：对每个点，累加所有障碍的 repulsion 向量。
    返回 (N,2)。
    """
    if not obstacles or P.size == 0:
        return np.zeros_like(P)
    A = np.zeros_like(P)
    for i in range(P.shape[0]):
        acc = np.zeros(2)
        pi = P[i]
        for ob in obstacles:
            acc += ob.vector_away(pi, wrap, W)
        A[i] = acc
    return A


def project_out_of_obstacles(P: np.ndarray, V: np.ndarray, W: World):
    """
    硬投影：若粒子落入障碍内部，则把它推至障碍边界外，并反射法向速度分量。
    注意：为简洁与稳定，Circle 处理精确，Polygon 用“向质心外”近似。
    """
    if not W.hard_obstacle or not getattr(W, "obstacles", None) or P.size == 0:
        return

    for i in range(P.shape[0]):
        p = P[i]
        v = V[i]
        for ob in W.obstacles:
            if isinstance(ob, CircleObstacle):
                dx = p[0] - ob.cx
                dy = p[1] - ob.cy
                # wrap 环境下找最近镜像
                if W.boundary == "wrap":
                    if dx > 0.5*W.width:  dx -= W.width
                    if dx < -0.5*W.width: dx += W.width
                    if dy > 0.5*W.height: dy -= W.height
                    if dy < -0.5*W.height: dy += W.height
                d = math.sqrt(dx*dx + dy*dy) + 1e-12
                if d < ob.r:
                    # 推至圆外（留出 1e-4 的间隙避免反复投影）
                    n = np.array([dx/d, dy/d])
                    p_new = np.array([ob.cx, ob.cy]) + n * (ob.r + 1e-4)
                    # 反射速度的法向分量
                    vn = np.dot(v, n) * n
                    vt = v - vn
                    v_new = vt - vn * 0.6   # 带阻尼的反射
                    # 写回
                    P[i] = p_new
                    V[i] = v_new

            elif isinstance(ob, PolyObstacle):
                # 近似：inside 则沿“质心->点”的法向推出
                # 更精确可做 point-to-segment 投影得到最近点与法线
                def _inside(x, y):
                    c = False
                    n = len(ob.pts)
                    for k in range(n):
                        x1, y1 = ob.pts[k]
                        x2, y2 = ob.pts[(k+1) % n]
                        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1 + 1e-12) + x1):
                            c = not c
                    return c

                if _inside(float(p[0]), float(p[1])):
                    c = np.array([sum(x for x, _ in ob.pts)/len(ob.pts),
                                  sum(y for _, y in ob.pts)/len(ob.pts)], dtype=float)
                    d = p - c
                    nrm = np.linalg.norm(d) + 1e-12
                    n = d / nrm
                    # 推出一个小步长（把点放到多边形外的小距离处）
                    p_new = p + n * 1e-3
                    vn = np.dot(v, n) * n
                    vt = v - vn
                    v_new = vt - vn * 0.6
                    P[i] = p_new
                    V[i] = v_new

# ======================= Spatial Hash Grid =======================

class SpatialGrid:
    def __init__(self, width: float, height: float, cell: float, boundary: str = "wrap"):
        self.w = width; self.h = height; self.s = cell
        self.nx = max(1, int(math.ceil(width / cell)))
        self.ny = max(1, int(math.ceil(height / cell)))
        self.boundary = boundary
        self.cells: Dict[Tuple[int,int], List[int]] = {}

    def clear(self):
        self.cells.clear()

    def cell_index(self, x: np.ndarray, y: np.ndarray):
        ix = np.floor(x / self.s).astype(int) % self.nx
        iy = np.floor(y / self.s).astype(int) % self.ny
        return ix, iy

    def insert_all(self, P: np.ndarray):
        self.clear()
        if P.size == 0: return
        ix, iy = self.cell_index(P[:,0], P[:,1])
        for i in range(P.shape[0]):
            self.cells.setdefault((int(ix[i]), int(iy[i])), []).append(i)

    def neighbors(self, P: np.ndarray, i: int, radius: float) -> List[int]:
        x, y = P[i]; r = radius
        cx, cy = self.cell_index(np.array([x]), np.array([y]))
        cx, cy = int(cx[0]), int(cy[0])
        out: List[int] = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                nx = (cx+dx) % self.nx; ny = (cy+dy) % self.ny
                for j in self.cells.get((nx,ny), []):
                    if j == i: continue
                    dx_ = P[j,0]-x; dy_ = P[j,1]-y
                    if self.boundary == "wrap":
                        if dx_ > 0.5*self.w: dx_ -= self.w
                        if dx_ < -0.5*self.w: dx_ += self.w
                        if dy_ > 0.5*self.h: dy_ -= self.h
                        if dy_ < -0.5*self.h: dy_ += self.h
                    if dx_*dx_ + dy_*dy_ <= r*r:
                        out.append(j)
        return out

# ======================= Utility =======================

def limit_speed(V: np.ndarray, vmin: float, vmax: float):
    s = np.linalg.norm(V, axis=1) + 1e-12
    scale = np.clip(s, vmin, vmax) / s
    V *= scale[:,None]

# ======================= Boid School =======================

@dataclass
class BoidParams:
    vmax: float = 0.015
    vmin: float = 0.003
    noise: float = 0.0004
    r_sep: float = 0.02
    r_ali: float = 0.06
    r_coh: float = 0.08
    w_sep: float = 1.5
    w_ali: float = 0.08
    w_coh: float = 0.06
    # split/merge
    target_group_size: int = 80
    merge_dist: float = 0.12
    split_jitter: float = 0.003

class BoidSchool:
    def __init__(self, n: int, world: World, prm: BoidParams, seed=0):
        self.n = n
        self.world = world
        self.prm = prm
        self.rng = np.random.default_rng(seed)
        self.P = self.rng.random((n,2)) * np.array([world.width, world.height])
        theta = self.rng.uniform(0, 2*math.pi, n)
        speed0 = self.rng.uniform(prm.vmin, prm.vmax, n)
        self.V = np.stack([np.cos(theta), np.sin(theta)], axis=1) * speed0[:,None]
        self.grid = SpatialGrid(world.width, world.height, world.cell, world.boundary)
        self.goals: List[Tuple[float,float,float]] = []  # (x,y,w)
        # group labels for split/merge
        g = max(1, n // prm.target_group_size)
        self.labels = self.rng.integers(0, g, size=n)

        # legacy per-school obstacles (仍兼容)
        self.circles: List[CircleObstacle] = []
        self.polys: List[PolyObstacle] = []

        # stats
        self.time = 0.0
        self.heat_bins = (128, 128)
        self.heat = np.zeros(self.heat_bins, dtype=np.float64)

    def set_goals(self, goals: List[Tuple[float,float,float]]): self.goals = goals
    def add_circle_obstacles(self, circles: List[CircleObstacle]): self.circles.extend(circles)
    def add_poly_obstacles(self, polys: List[PolyObstacle]): self.polys.extend(polys)

    def _wrap_delta(self, d: Vec) -> Vec:
        if self.world.boundary != "wrap": return d
        if d[0] > 0.5*self.world.width:  d[0] -= self.world.width
        if d[0] < -0.5*self.world.width: d[0] += self.world.width
        if d[1] > 0.5*self.world.height: d[1] -= self.world.height
        if d[1] < -0.5*self.world.height: d[1] += self.world.height
        return d

    def _group_centers(self) -> Dict[int, Vec]:
        centers: Dict[int, Vec] = {}
        for lab in np.unique(self.labels):
            idx = np.where(self.labels == lab)[0]
            if len(idx) == 0: continue
            centers[lab] = np.mean(self.P[idx], axis=0)
        return centers

    def _split_merge_update(self):
        prm = self.prm
        centers = self._group_centers()
        # merge
        labs = list(centers.keys())
        merged = {}
        for i in range(len(labs)):
            for j in range(i+1, len(labs)):
                ci, cj = centers[labs[i]], centers[labs[j]]
                d = self._wrap_delta(cj - ci)
                if d @ d < prm.merge_dist*prm.merge_dist:
                    merged[labs[j]] = labs[i]
        if merged:
            for j, i in merged.items():
                self.labels[self.labels == j] = i
        # split
        for lab in np.unique(self.labels):
            idx = np.where(self.labels == lab)[0]
            if len(idx) > 1.8*prm.target_group_size:
                dirv = self.rng.normal(size=2)
                dirv /= (np.linalg.norm(dirv)+1e-9)
                proj = (self.P[idx] @ dirv)
                med = np.median(proj)
                left = idx[proj <= med]; right = idx[proj > med]
                newlab = self.labels.max() + 1
                self.labels[right] = newlab
                self.V[left]  -= dirv * prm.split_jitter
                self.V[right] += dirv * prm.split_jitter

    def _accumulate_heat(self):
        x = np.clip(self.P[:,0] / self.world.width, 0, 0.9999)
        y = np.clip(self.P[:,1] / self.world.height, 0, 0.9999)
        ix = (x * self.heat_bins[0]).astype(int)
        iy = (y * self.heat_bins[1]).astype(int)
        np.add.at(self.heat, (ix, iy), 1)

    def step(self, predators: Optional[np.ndarray]=None, dt: float=1.0):
        prm = self.prm; W = self.world
        P, V = self.P, self.V
        N = P.shape[0]
        self.grid.insert_all(P)
        acc = np.zeros_like(V)

        centers = self._group_centers()

        # 构建障碍列表（World 全局 + School 层）
        obs_list: List[Obstacle] = []
        if getattr(W, "obstacles", None): obs_list.extend(W.obstacles)
        if self.circles: obs_list.extend(self.circles)
        if self.polys:   obs_list.extend(self.polys)

        # 软避障（预先算好）
        obs_acc = obstacle_repulsion(P, obs_list, W.boundary == "wrap", W)  # (N,2)

        for i in range(N):
            n_sep = self.grid.neighbors(P, i, prm.r_sep)
            n_ali = self.grid.neighbors(P, i, prm.r_ali)
            n_coh = self.grid.neighbors(P, i, prm.r_coh)

            fsep = np.zeros(2)
            for j in n_sep:
                d = P[j] - P[i]
                d = self._wrap_delta(d)
                dist2 = d[0]*d[0]+d[1]*d[1] + 1e-12
                fsep -= d / dist2

            fali = np.zeros(2)
            if n_ali: fali = np.mean(V[n_ali], axis=0) - V[i]

            fcoh = np.zeros(2)
            if n_coh:
                center = np.mean(P[n_coh], axis=0)
                d = self._wrap_delta(center - P[i]); fcoh = d

            # group center attraction (light)
            lab = self.labels[i]
            if lab in centers:
                dgc = self._wrap_delta(centers[lab] - P[i])
                fcoh += 0.4 * dgc

            a = prm.w_sep*fsep + prm.w_ali*fali + prm.w_coh*fcoh

            # goals
            for (gx,gy,w) in self.goals:
                d = self._wrap_delta(np.array([gx,gy]) - P[i])
                a += w * d

            # predator avoidance
            if predators is not None and predators.size > 0:
                for k in range(predators.shape[0]):
                    d = self._wrap_delta(P[i] - predators[k])
                    dist2 = d[0]*d[0] + d[1]*d[1]
                    if dist2 < 0.25*0.25:
                        a += 0.14 * d / (dist2 + 1e-9)

            # flow field drift
            if W.flow is not None:
                ux, uy = W.flow(P[i,0], P[i,1], self.time)
                a += np.array([ux, uy]) * 0.3

            # 软避障力
            a += obs_acc[i]

            # noise
            a += (self.rng.random(2) - 0.5) * prm.noise
            acc[i] = a

        V += acc * dt
        limit_speed(V, prm.vmin, prm.vmax)
        P += V * dt

        # 边界
        if W.boundary == "wrap":
            P[:,0] = np.mod(P[:,0], W.width)
            P[:,1] = np.mod(P[:,1], W.height)
        else:
            for d, L in ((0,W.width),(1,W.height)):
                over = P[:,d] > L; P[over,d] = 2*L - P[over,d]; V[over,d] *= -1
                under = P[:,d] < 0; P[under,d] = -P[under,d]; V[under,d] *= -1

        # 硬投影（避免穿障碍）
        project_out_of_obstacles(P, V, W)

        # bookkeeping
        self.time += dt
        self._accumulate_heat()
        self._split_merge_update()

# ======================= Predator (Whale) =======================

@dataclass
class PredatorParams:
    vmax: float = 0.025
    vmin: float = 0.006
    accel: float = 0.001
    sense: float = 0.25
    capture: float = 0.02

class Predator:
    def __init__(self, world: World, prm: PredatorParams, seed=1):
        self.world = world; self.prm = prm
        self.rng = np.random.default_rng(seed)
        self.P = self.rng.random((1,2)) * np.array([world.width, world.height])
        theta = self.rng.uniform(0,2*math.pi)
        self.V = np.array([[math.cos(theta), math.sin(theta)]]) * prm.vmin

    def step(self, preyP: np.ndarray, dt: float=1.0):
        W = self.world; prm = self.prm
        P = self.P; V = self.V
        target = None; bestd2 = None
        for j in range(preyP.shape[0]):
            d = preyP[j] - P[0]
            if W.boundary == "wrap":
                if d[0] > 0.5*W.width:  d[0] -= W.width
                if d[0] < -0.5*W.width: d[0] += W.width
                if d[1] > 0.5*W.height: d[1] -= W.height
                if d[1] < -0.5*W.height: d[1] += W.height
            d2 = d[0]*d[0] + d[1]*d[1]
            if d2 <= prm.sense*prm.sense and (bestd2 is None or d2 < bestd2):
                bestd2 = d2; target = d

        a = np.zeros(2)
        if target is not None:
            desired = target / (math.sqrt(bestd2) + 1e-9) * prm.vmax
            steer = desired - V[0]
            s = np.linalg.norm(steer) + 1e-12
            steer *= min(prm.accel, s) / s
            a += steer
        else:
            a += (self.rng.random(2) - 0.5) * prm.accel*0.5

        V[0] += a * dt
        limit_speed(V, prm.vmin, prm.vmax)
        P[0] += V[0] * dt

        # boundary
        if W.boundary == "wrap":
            P[:,0] = np.mod(P[:,0], W.width)
            P[:,1] = np.mod(P[:,1], W.height)
        else:
            for d, L in ((0,W.width),(1,W.height)):
                if P[0,d] > L: P[0,d] = 2*L - P[0,d]; V[0,d] *= -1
                if P[0,d] < 0: P[0,d] = -P[0,d]; V[0,d] *= -1

    def capture_indices(self, preyP: np.ndarray) -> List[int]:
        prm = self.prm; W = self.world
        out = []
        for j in range(preyP.shape[0]):
            d = preyP[j] - self.P[0]
            if W.boundary == "wrap":
                if d[0] > 0.5*W.width:  d[0] -= W.width
                if d[0] < -0.5*W.width: d[0] += W.width
                if d[1] > 0.5*W.height: d[1] -= W.height
                if d[1] < -0.5*W.height: d[1] += W.height
            if d[0]*d[0] + d[1]*d[1] <= prm.capture*prm.capture:
                out.append(j)
        return out