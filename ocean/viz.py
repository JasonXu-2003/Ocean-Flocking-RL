# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu

#!/usr/bin/env python3
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from .core import CircleObstacle, PolyObstacle  # 兼容：用于 isinstance 判断（可选）

# ---------------- 基础工具 ----------------

def has_ffmpeg() -> bool:
    from matplotlib.animation import writers
    return ('ffmpeg' in writers.list()) or (shutil.which('ffmpeg') is not None)

def _ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _save_anim(ani: FuncAnimation, save: str, interval: int, dpi: int = 180):
    """保存动画：优先 mp4（需要 ffmpeg），否则回退 gif。"""
    fps = max(1, int(1000 / max(1, interval)))
    if not save:
        plt.show()
        return

    ext = os.path.splitext(save)[1].lower()
    _ensure_dir_for(save)

    if ext == ".mp4":
        if has_ffmpeg():
            ani.save(save, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=dpi)
        else:
            fallback = os.path.splitext(save)[0] + ".gif"
            print(f"[ocean.viz] ffmpeg not found -> fallback to {fallback}")
            ani.save(fallback, writer=PillowWriter(fps=fps))
    elif ext == ".gif":
        ani.save(save, writer=PillowWriter(fps=fps))
    else:
        print(f"[ocean.viz] unknown file extension: {ext}; showing preview instead.")
        plt.show()

# ---------------- 绘制工具 ----------------

def _draw_framebox(ax, W):
    ax.set_xlim(0, getattr(W, "width", 1.0))
    ax.set_ylim(0, getattr(W, "height", 1.0))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((0, 0), getattr(W, "width", 1.0), getattr(W, "height", 1.0),
                               fill=False, linewidth=1))

def _draw_obstacles_from_world(ax, W):
    """优先从 W.obstacles 绘制（新接口），若无则尝试 school.circles/polys 的旧接口在 draw_world 中处理。"""
    obs = getattr(W, "obstacles", None)
    if not obs:
        return
    for ob in obs:
        # 兼容 CircleObstacle/PolyObstacle，也兼容 dict 格式（如果你用 JSON 直接传）
        if isinstance(ob, CircleObstacle) or getattr(ob, "r", None) is not None:
            x = getattr(ob, "x", None) or getattr(ob, "cx", None)
            y = getattr(ob, "y", None) or getattr(ob, "cy", None)
            r = getattr(ob, "r", None)
            if x is not None and y is not None and r is not None:
                ax.add_patch(plt.Circle((float(x), float(y)), float(r),
                                        fill=False, lw=2, ls="--", ec="red"))
        elif isinstance(ob, PolyObstacle) or getattr(ob, "pts", None) is not None:
            pts = np.asarray(getattr(ob, "pts", []), dtype=float)
            if pts.size >= 4:
                ax.plot(np.r_[pts[:, 0], pts[0, 0]],
                        np.r_[pts[:, 1], pts[0, 1]], "r--", lw=2)

def draw_world(ax, W, school=None):
    """画边框 + 障碍。school 仅作旧接口兼容：school.circles / school.polys。"""
    _draw_framebox(ax, W)
    # 新接口：World.obstacles
    _draw_obstacles_from_world(ax, W)

    # 旧接口兼容：school.circles / school.polys（如果还在旧的 BoidSchool 里保存）
    if school is not None:
        circles = getattr(school, "circles", [])
        for c in circles or []:
            cx = getattr(c, "cx", getattr(c, "x", None))
            cy = getattr(c, "cy", getattr(c, "y", None))
            r = getattr(c, "r", None)
            if cx is not None and cy is not None and r is not None:
                ax.add_patch(plt.Circle((float(cx), float(cy)), float(r),
                                        fill=False, lw=1.5, ec="red"))
        polys = getattr(school, "polys", [])
        for poly in polys or []:
            pts = np.asarray(getattr(poly, "pts", []), dtype=float)
            if pts.size >= 4:
                ax.plot(np.r_[pts[:, 0], pts[0, 0]],
                        np.r_[pts[:, 1], pts[0, 1]], "r--", lw=1.5)

def draw_schools(ax, schools, colors=None):
    """多鱼群散点绘制。"""
    if colors is None:
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, sch in enumerate(schools):
        P = getattr(sch, "P", None)
        if isinstance(P, np.ndarray) and P.size > 0:
            ax.scatter(P[:, 0], P[:, 1], s=8, c=colors[i % len(colors)], alpha=0.95)

# ---------------- 动画：鲸捕食 ----------------

def animate_whales(prey_school, whales, steps=1500, interval=20, save=None, stats=None):
    """
    仍按原签名：prey_school 为单群；如你想支持多鱼群，建议外层把多个 school 合并成一个视图或改签名。
    """
    W = getattr(prey_school, "world", None)
    if W is None:
        raise ValueError("prey_school.world is required")

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_world(ax, W, prey_school)

    # 鱼群散点
    P0 = getattr(prey_school, "P", np.zeros((0, 2)))
    fish = ax.scatter(P0[:, 0], P0[:, 1], s=8)

    # 鲸位置
    if whales:
        WXY = np.vstack([w.P for w in whales])
    else:
        WXY = np.zeros((0, 2))
    whale_sc = ax.scatter(WXY[:, 0] if WXY.size else [],
                          WXY[:, 1] if WXY.size else [],
                          s=160, marker='>', edgecolors='black',
                          facecolors='none', linewidths=1.5)

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')
    eaten_total = [0]

    def update(frame):
        # whales move first
        for w in whales or []:
            w.step(prey_school.P)

        # fish react
        preds = np.vstack([w.P for w in whales]) if whales else None
        prey_school.step(predators=preds)

        # capture
        removed = []
        for w in whales or []:
            idx = w.capture_indices(prey_school.P)
            if idx:
                removed.extend(idx)
        if removed:
            removed = np.unique(np.array(removed, dtype=int))
            prey_school.P = np.delete(prey_school.P, removed, axis=0)
            prey_school.V = np.delete(prey_school.V, removed, axis=0)
            if hasattr(prey_school, "labels") and getattr(prey_school, "labels", None) is not None and len(prey_school.labels) > 0:
                prey_school.labels = np.delete(prey_school.labels, removed, axis=0)
            eaten_total[0] += len(removed)

        # 更新绘图
        fish.set_offsets(prey_school.P)
        if whales:
            whale_sc.set_offsets(np.vstack([w.P for w in whales]))
        lbls = getattr(prey_school, "labels", None)
        n_groups = len(np.unique(lbls)) if isinstance(lbls, np.ndarray) and lbls.size > 0 else 1
        title.set_text(f"Frame {frame} | Fish: {prey_school.P.shape[0]} | Eaten: {eaten_total[0]} | Groups: {n_groups}")

        # 统计
        if stats is not None and hasattr(stats, "log"):
            t_now = getattr(prey_school, "time", frame)
            V_now = getattr(prey_school, "V", None)
            stats.log(t_now, prey_school.P.shape[0], eaten_total[0], V_now)

        return fish, whale_sc, title

    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=False)
    _save_anim(ani, save, interval)

# ---------------- 动画：仅目标/流场 ----------------

def animate_goals(prey_school, steps=1500, interval=20, save=None, stats=None):
    W = getattr(prey_school, "world", None)
    if W is None:
        raise ValueError("prey_school.world is required")

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_world(ax, W, prey_school)

    P0 = getattr(prey_school, "P", np.zeros((0, 2)))
    fish = ax.scatter(P0[:, 0], P0[:, 1], s=8)

    goals = getattr(prey_school, "goals", None)
    if goals:
        goal_pts = np.array([[g[0], g[1]] for g in goals], dtype=float)
        gl = ax.scatter(goal_pts[:, 0], goal_pts[:, 1], s=40, marker='x')
    else:
        goal_pts = np.zeros((0, 2))
        gl = None

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

    def update(frame):
        prey_school.step(predators=None)
        fish.set_offsets(prey_school.P)
        if gl is not None:
            gl.set_offsets(goal_pts)

        lbls = getattr(prey_school, "labels", None)
        n_groups = len(np.unique(lbls)) if isinstance(lbls, np.ndarray) and lbls.size > 0 else 1
        title.set_text(f"Frame {frame} | Fish: {prey_school.P.shape[0]} | Groups: {n_groups}")

        if stats is not None and hasattr(stats, "log"):
            t_now = getattr(prey_school, "time", frame)
            V_now = getattr(prey_school, "V", None)
            stats.log(t_now, prey_school.P.shape[0], 0, V_now)

        return (fish,) if gl is None else (fish, gl, title)

    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=False)
    _save_anim(ani, save, interval)