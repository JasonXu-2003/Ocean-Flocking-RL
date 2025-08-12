# -*- coding: utf-8 -*-
# 创作日期: 2025年8月9日
# 作者: JasonXu

#!/usr/bin/env python3
import argparse
from ocean.scenarios.goals_flow import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fish", type=int, default=300)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--mp4", type=str, default=None)
    ap.add_argument("--gif", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    save = os.makedirs(os.path.dirname(args.mp4) or ".", exist_ok=True)
args.mp4 or os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
args.gif
    run(n_fish=args.n_fish, steps=args.steps, save=save, seed=args.seed)
