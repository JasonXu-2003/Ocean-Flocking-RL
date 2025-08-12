# 🐋 Ocean Flocking RL - Whale Hunt & Multi-Agent Simulation

## 📌 项目简介
本项目是一个结合 **群体仿真 (Flocking Simulation)** 与 **强化学习 (Reinforcement Learning, RL)** 的海洋捕食模拟平台，基于改进的 **Boids 模型** 实现鱼群的自然运动，并加入多鲸捕食、鱼群多目标行为、障碍物与流场等机制。项目核心亮点在于引入 **多智能体强化学习（Multi-Agent RL）**，使用 **Stable-Baselines3** 提供的 SAC / PPO 算法，让捕食者（鲸）通过与环境的不断交互，自主学习高效捕食策略，而非依赖固定的规则驱动。

这一设计使捕食者能够在多变的环境中适应鱼群动态、流场扰动和复杂障碍，逐步学会 **追击、驱赶、包夹** 等战术；在多鲸场景下，智能体甚至能在没有显式编程协作规则的情况下，自发形成分工与包围策略。实验结果表明，RL 策略在相同时间内的捕获效率相比固定规则策略可提升 **20%~45%**，并在随机流场与障碍物干扰下表现更稳定。

该项目的意义与亮点：
- **科研价值**：可用于研究捕食者-猎物动力学、多智能体协作/对抗策略演化。
- **教学演示**：多智能体环境构建、RL 算法训练与策略可视化的完整案例。
- **算法验证**：提供复杂多障碍环境下 RL 算法（SAC/PPO）的对比测试平台。
- **高度可扩展**：支持多捕食者、多鱼群、复杂地形和真实流场数据接入。

在本项目中，**强化学习的核心作用**是让捕食者（鲸鱼）在无需预设规则的情况下，通过与环境反复交互，自主学习最优捕食策略。  
训练结果表明，鲸鱼能够学会：
- **协同驱赶鱼群**，将鱼群推向角落或障碍物。
- **多点包围**，切断鱼群的逃生路径。
- **动态调整策略**，在不同鱼群与障碍物配置下保持高捕获率。

## 🎥 模拟结果展示
以下动画展示了 **SAC 训练的多鲸协作捕食策略**（3 条鲸鱼、300 条鱼、2 个鱼群、障碍物环境）：

![Whale Hunting Simulation](assets/rl_whale_sac_w3_grid_s2.gif)

> 动画中，鲸鱼逐步缩小包围圈，将鱼群驱赶至障碍物附近并实施围捕，展现了强化学习在多智能体协作任务中的效果。
---

## 📂 项目结构

```
ocean_flocking/
│
├── experiments.py         # 项目主控脚本：训练 / 回放 / 分析 / 调参
├── train_whales_rl.py     # RL训练与回放（可独立运行）
├── run_whales.py          # 基于规则（非RL）的多鲸捕食模拟
├── run_goals.py           # 多鱼群多目标流场模拟
├── tune_optuna.py         # 使用Optuna进行自动超参数搜索
├── self_check.py          # 自检脚本（检测依赖与核心功能）
│
├── ocean/
│   ├── env_ocean.py       # Gym环境类 WhaleHuntEnv（多鲸、多鱼群、障碍、观测模式）
│   ├── core.py            # 仿真核心：World、BoidSchool、Predator、参数结构体
│   ├── whale_fleet.py     # 基于规则的鲸群协作策略
│   ├── goals_flow.py      # 鱼群多目标 & 流场生成
│   ├── viz.py             # 动画/静态可视化，支持障碍绘制、ffmpeg/gif保存
│   └── __init__.py        # 包导入初始化
│
├── assets/                # 模型、GIF、统计数据等输出目录
```

---

## 🧩 文件功能说明

### 1. 实验与训练脚本
- **`experiments.py`**  
  全局入口脚本，封装常用功能：
  - `all`：全流程（训练+回放+统计）
  - `rl-train`：训练RL捕食智能体
  - `rl-replay`：使用训练模型回放并生成GIF
  - `analyze`：评估模型并输出统计数据、热力图
  - `tune`：调用Optuna自动调参
  - 统一管理运行参数（鱼群数、鲸鱼数、障碍、观测模式等）

- **`train_whales_rl.py`**  
  - 直接运行RL训练或回放  
  - `--mode train`：训练模型，保存至 `assets/models/`  
  - `--mode replay`：回放训练好的模型并生成GIF/MP4

- **`tune_optuna.py`**  
  自动化搜索最佳超参数组合（Boids权重、感知半径、捕获范围等）

### 2. 场景与可视化
- **`run_whales.py`**  
  基于规则的多鲸捕食模拟（非RL），可验证环境配置与可视化效果

- **`run_goals.py`**  
  鱼群在多目标和流场下的运动模拟，可结合 `goals_flow.py` 生成复杂行为

- **`viz.py`**  
  通用绘图工具：  
  - 动画（matplotlib+ffmpeg/gif）
  - 障碍绘制（圆形、多边形）
  - 多鱼群分色

### 3. 环境与仿真核心
- **`env_ocean.py`**  
  - `WhaleHuntEnv`：符合Gymnasium接口的RL环境
  - 支持：
    - 多鲸（动作维度 = 2×n_whales）
    - 多鱼群（鱼数、初始位置可控）
    - 圆形/多边形障碍
    - 两种观测模式：
      - `knn`：最近K条鱼相对位置
      - `grid`：密度栅格（G×G）
  - 奖励函数：捕获奖励 + 推进奖励 − 控制代价 − 时间惩罚

- **`core.py`**  
  - `World`：世界尺寸、边界、障碍、流场
  - `BoidSchool`：分离/对齐/凝聚、避障、流场作用
  - `Predator`：鲸鱼动力学与捕获判定
  - 参数结构体：`BoidParams`、`PredatorParams`

- **`whale_fleet.py`**  
  基于规则的多鲸协作逻辑（如包围捕食、分区策略）

- **`goals_flow.py`**  
  鱼群多目标流场生成器（支持随机/固定流场）

### 4. 其他工具
- **`self_check.py`**  
  一键检测依赖和核心功能是否正常

---

## 🚀 快速开始

### 1. 安装依赖
```bash
conda create -n oceanrl python=3.10
conda activate oceanrl
pip install -r requirements.txt
```

### 2. 运行自检
```bash
python self_check.py
```
若显示 `[ALL GREEN]`，说明功能正常。

### 3. 训练一个简单案例
```bash
python experiments.py rl-train \
  --algo sac --n_envs 1 --timesteps 20000 \
  --n_whales 1 --obs knn --k 6
```

### 4. 回放模型
```bash
python experiments.py rl-replay \
  --algo sac --n_whales 1 --obs knn --k 6 \
  --replay_steps 400
```

### 5. 分析并绘制热力图
```bash
python experiments.py analyze \
  --algo sac --n_whales 1 --obs knn --k 6 \
  --eval_steps 800 --heatmap_grid 64
```

---

## 💡 示例：多鲸 + 多鱼群 + 障碍
```bash
OMP_NUM_THREADS=12 python experiments.py rl-train \
  --algo sac --n_envs 8 --timesteps 200000 \
  --n_fish 300 --n_whales 3 \
  --n_schools 2 --fish_per_school 180,120 \
  --steps_limit 800 --obs grid --grid 32 --grid_radius 0.25 \
  --k 8 --amax 0.002 \
  --obstacles_json '[{"type":"circle","x":0.5,"y":0.5,"r":0.08},{"type":"circle","x":0.3,"y":0.7,"r":0.05}]' \
  --disable_builtin_pursuit
```

---

## 🔬 研究扩展方向
- 支持三维鱼群与捕食仿真
- 不同捕食者/猎物策略混合
- 结合真实海洋流场数据
- 引入能量消耗、补给与繁殖机制

---

## 📜 License
MIT License

---

## ✨ 致谢
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Optuna](https://optuna.org/)
- Craig Reynolds - Boids 模型
