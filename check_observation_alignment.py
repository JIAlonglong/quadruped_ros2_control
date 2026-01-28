#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
观测（Observation）对齐检查程序
--------------------------------
检查三端是否一致：
  - 训练端：Isaac Gym legged_robot.py 的 obs_buf 前 53 维（proprio）
  - 部署 C++：StateRL.cpp 构建的 proprio
  - 部署 Python：Extreme-Parkour-Onboard unitree_ros2_real.get_proprio()

包含：
  1) 观测维度布局说明（每一维的含义、缩放、顺序）
  2) 关节/足端顺序与 reindex 是否与训练端一致
  3) 用「同一份虚构原始数据」分别按 C++ 规则和 Python/训练规则构建观测，对比是否一致
"""

import json
import os
import sys

try:
    import yaml
except ImportError:
    print("请安装 PyYAML: pip install pyyaml")
    sys.exit(1)

# 颜色
class C:
    H = '\033[95m\033[1m'
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    E = '\033[0m'

def header(t): print(f"\n{C.H}=== {t} ==={C.E}")
def ok(t):    print(f"{C.G}[✔] {t}{C.E}")
def warn(t):  print(f"{C.Y}[!] {t}{C.E}")
def fail(t):  print(f"{C.R}[✘] {t}{C.E}")

# -----------------------------------------------------------------------------
# 1) 观测布局规范（与 legged_robot.py obs_buf 前 53 维一致）
# -----------------------------------------------------------------------------
PROPRIO_SPEC = [
    (0,   3,  "base_ang_vel",       "角速度(机体系) * ang_vel_scale", "gyro[0:3]"),
    (3,   5,  "imu_obs",            "roll, pitch（弧度）", "quat->RPY"),
    (5,   8,  "yaw_info",           "0, delta_yaw, delta_next_yaw", "目标偏航相关，部署常为 0,0,0"),
    (8,  11,  "commands",           "0, 0, forward_cmd（前向速度指令）", "未乘 lin_vel_scale"),
    (11, 13,  "env_class",          "(env!=17), (env==17)，部署常用 [1,0]", "parkour/walk 等"),
    (13, 25,  "dof_pos",            "(pos - default) * dof_pos_scale，顺序为策略序 [FR,FL,RR,RL]", "重排后 12 维"),
    (25, 37,  "dof_vel",            "dof_vel * dof_vel_scale，策略序 [FR,FL,RR,RL]", "重排后 12 维"),
    (37, 49,  "last_actions",       "上一帧动作，策略序 [FR,FL,RR,RL]", "重排后 12 维"),
    (49, 53,  "contact",            "接触 (float) -0.5 或 0.5，顺序 [FR,FL,RR,RL]", "重排后 4 维"),
]

def build_observation_cpp_style(raw, cfg):
    """按 C++ StateRL 的公式，用 raw 里的“ROS 顺序”数据，构建 53 维 proprio。"""
    import numpy as np
    dof_reindex = cfg.get("dof_reindex", list(range(12)))
    feet_reindex = cfg.get("feet_reindex", [0, 1, 2, 3])
    # raw 约定：joint_pos_ros, joint_vel_ros, last_action_ros 均为长度 12，ROS 顺序 [FL,FR,RL,RR]
    #           foot_force_ros 长度 4，ROS 顺序 [FL,FR,RL,RR]
    q_ros = np.array(raw["joint_pos_ros"], dtype=np.float32)
    dq_ros = np.array(raw["joint_vel_ros"], dtype=np.float32)
    last_ros = np.array(raw["last_action_ros"], dtype=np.float32)
    default_ros = np.array(raw["default_dof_pos_ros"], dtype=np.float32)
    foot_ros = np.array(raw["foot_force_ros"], dtype=np.float32)
    thr = float(cfg.get("foot_force_threshold", 25.0))

    ang_vel = np.array(raw["ang_vel"], dtype=np.float32) * float(cfg.get("ang_vel_scale", 0.25))
    imu = np.array(raw["imu_rp"], dtype=np.float32)  # [roll, pitch]
    yaw_info = np.array(raw.get("yaw_info", [0, 0, 0]), dtype=np.float32)
    commands = np.array(raw.get("commands", [0, 0, 0.5]), dtype=np.float32)
    env_class = np.array(raw.get("env_class", [1, 0]), dtype=np.float32)

    # 关节：ROS → policy
    q_policy = q_ros[dof_reindex]
    dq_policy = dq_ros[dof_reindex]
    last_policy = last_ros[dof_reindex]
    default_policy = default_ros[dof_reindex]
    dof_pos_scale = float(cfg.get("dof_pos_scale", 1.0))
    dof_vel_scale = float(cfg.get("dof_vel_scale", 0.05))
    dof_pos = (q_policy - default_policy) * dof_pos_scale
    dof_vel = dq_policy * dof_vel_scale

    # 接触：ROS 顺序，标量阈值 -> 再 reindex 到 policy
    contact_ros = np.where(foot_ros > thr, 0.5, -0.5)
    contact = contact_ros[feet_reindex]

    parts = [ang_vel, imu, yaw_info, commands, env_class, dof_pos, dof_vel, last_policy, contact]
    return np.concatenate(parts).astype(np.float32)

def build_observation_python_style(raw, cfg_py):
    """按 Python unitree_ros2_real 的 get_proprio：假设 dof/feet 已是「仿真/策略序」[FR,FL,RR,RL]。"""
    import numpy as np
    scales = cfg_py.get("normalization", {}).get("obs_scales", {})
    ang_vel = np.array(raw["ang_vel"], dtype=np.float32) * float(scales.get("ang_vel", 0.25))
    imu = np.array(raw["imu_rp"], dtype=np.float32)
    yaw_info = np.array(raw.get("yaw_info", [0, 0, 0]), dtype=np.float32)
    vx = float(raw.get("commands", [0, 0, 0.5])[2])
    commands = np.array([0, 0, vx], dtype=np.float32)
    env_class = np.array(raw.get("env_class", [1, 0]), dtype=np.float32)
    # Python 端 dof 已是 [FR,FL,RR,RL]，且 default 也是该顺序
    q = np.array(raw["joint_pos_policy"], dtype=np.float32)
    default = np.array(raw["default_dof_pos_policy"], dtype=np.float32)
    dq = np.array(raw["joint_vel_policy"], dtype=np.float32)
    last = np.array(raw["last_action_policy"], dtype=np.float32)
    dof_pos = (q - default) * float(scales.get("dof_pos", 1.0))
    dof_vel = dq * float(scales.get("dof_vel", 0.05))
    contact = np.array(raw["contact_policy"], dtype=np.float32)  # 已为 [FR,FL,RR,RL]，-0.5/0.5
    parts = [ang_vel, imu, yaw_info, commands, env_class, dof_pos, dof_vel, last, contact]
    return np.concatenate(parts).astype(np.float32)

def main():
    ws = "/home/jialonglong/Quadruped_ws"
    path_cpp_yaml = os.path.join(
        ws, "src/quadruped_ros2_control/descriptions/peak_lab/depth_go2_description/config/extreme_parkour_common/config.yaml"
    )
    path_py_json = os.path.join(
        ws, "external/extreme-parkour/external/original_parkour/Extreme-Parkour-Onboard/traced/config.json"
    )

    header("观测（Observation）对齐检查 —— 与强化学习训练端 / 部署端")
    print("目标：确认 C++ 控制器构建的 proprio（53 维）与训练 / Python 部署 一致")
    print("C++ 配置:", path_cpp_yaml)
    print("Python/训练 配置:", path_py_json)

    with open(path_cpp_yaml, "r", encoding="utf-8") as f:
        cpp_cfg = yaml.safe_load(f)
    with open(path_py_json, "r", encoding="utf-8") as f:
        py_cfg = json.load(f)

    # ----- 1) 观测布局说明 -----
    header("1. 观测维度布局（53 维 proprio，与 legged_robot.py obs_buf 一致）")
    for start, end, name, desc, source in PROPRIO_SPEC:
        print(f"  [{start:2d}:{end:2d}] {name:16s}  {desc}  <- {source}")
    print("  策略序关节/足端均为 [FR, FL, RR, RL]（对应仿真 reindex / reindex_feet 之后）")

    # ----- 2) 关节与足端 reindex 是否与训练一致 -----
    header("2. 关节 / 足端顺序与训练端一致性")
    # 训练 legged_robot.py: reindex = [3,4,5, 0,1,2, 9,10,11, 6,7,8]  (sim FL,FR,RL,RR -> FR,FL,RR,RL)
    train_dof_reindex = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    train_feet_reindex = [1, 0, 3, 2]
    cpp_dof = cpp_cfg.get("dof_reindex") or list(range(12))
    cpp_feet = cpp_cfg.get("feet_reindex") or [0, 1, 2, 3]
    if cpp_dof == train_dof_reindex:
        ok("dof_reindex 与训练一致: [3,4,5, 0,1,2, 9,10,11, 6,7,8]（ROS [FL,FR,RL,RR] -> 策略 [FR,FL,RR,RL]）")
    else:
        fail(f"dof_reindex 与训练不一致: C++={cpp_dof}, 训练={train_dof_reindex}")
    if cpp_feet == train_feet_reindex:
        ok("feet_reindex 与训练一致: [1,0,3,2]（ROS [FL,FR,RL,RR] -> 策略 [FR,FL,RR,RL]）")
    else:
        fail(f"feet_reindex 与训练不一致: C++={cpp_feet}, 训练={train_feet_reindex}")
    warn("Python unitree_ros2_real 假设 low_state 的 motor_state 顺序已是 [FR,FL,RR,RL]，足端 foot_force[i] 顺序需与策略一致；若硬件为 [FL,FR,RL,RR]，需在 Python 端做与训练相同的 reindex。")

    # ----- 3) 缩放系数是否一致 -----
    header("3. 观测缩放与训练/部署一致")
    def get_nested(c, keys):
        for k in keys:
            c = c.get(k, {}) if isinstance(c, dict) else c
        return float(c) if isinstance(c, (int, float)) else 0.0
    for name, cpp_key, py_chain in [
        ("ang_vel_scale", "ang_vel_scale", ["normalization", "obs_scales", "ang_vel"]),
        ("dof_pos_scale", "dof_pos_scale", ["normalization", "obs_scales", "dof_pos"]),
        ("dof_vel_scale", "dof_vel_scale", ["normalization", "obs_scales", "dof_vel"]),
    ]:
        cv = float(cpp_cfg.get(cpp_key, 0))
        pv = get_nested(py_cfg, py_chain)
        if abs(cv - pv) < 1e-5:
            ok(f"{name}: C++/Python 均为 {cv}")
        else:
            fail(f"{name}: C++={cv} vs Python={pv}")

    # ----- 4) 用同一份逻辑状态，分别按 C++ 和 Python 规则构建，应得到相同 53 维 -----
    header("4. 同一原始数据下 C++ 规则 vs Python 规则 → 观测是否一致")
    # 构造“同一逻辑状态”：策略序 [FR,FL,RR,RL] 下 joint=[0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5], vel=0, last_action=0, contact=0.5
    policy_order = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
    ros_order = [policy_order[i] for i in [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]  # inv of train_dof_reindex
    raw_cpp = {
        "ang_vel": [0.0, 0.0, 0.0],
        "imu_rp": [0.0, 0.0],
        "yaw_info": [0.0, 0.0, 0.0],
        "commands": [0.0, 0.0, 0.5],
        "env_class": [1.0, 0.0],
        "joint_pos_ros": ros_order,
        "joint_vel_ros": [0.0] * 12,
        "last_action_ros": [0.0] * 12,
        "default_dof_pos_ros": ros_order,  # 与 joint_pos 同序
        "foot_force_ros": [50.0, 50.0, 50.0, 50.0],  # 全部接触
    }
    raw_py = {
        "ang_vel": [0.0, 0.0, 0.0],
        "imu_rp": [0.0, 0.0],
        "yaw_info": [0.0, 0.0, 0.0],
        "commands": [0.0, 0.0, 0.5],
        "env_class": [1.0, 0.0],
        "joint_pos_policy": policy_order,
        "joint_vel_policy": [0.0] * 12,
        "last_action_policy": [0.0] * 12,
        "default_dof_pos_policy": policy_order,
        "contact_policy": [0.5, 0.5, 0.5, 0.5],
    }
    obs_cpp = build_observation_cpp_style(raw_cpp, cpp_cfg)
    obs_py = build_observation_python_style(raw_py, py_cfg)
    diff = abs(obs_cpp - obs_py)
    max_diff = float(diff.max())
    if obs_cpp.shape != (53,) or obs_py.shape != (53,):
        fail("观测长度不是 53")
    elif max_diff < 1e-4:
        ok("在相同逻辑状态、相同缩放与 reindex 下，C++ 规则与 Python 规则得到的 53 维一致（max_diff < 1e-4）")
    else:
        fail(f"C++ 与 Python 规则得到的观测不一致，max_diff = {max_diff}")
        for i in range(53):
            if diff[i] >= 1e-4:
                print(f"    索引 {i} 差异: C++={obs_cpp[i]:.6f}  Python={obs_py[i]:.6f}")

    # ----- 5) 如何用「真实机器人数据」做逐帧对比 -----
    header("5. 用真实数据做观测逐帧对比（建议）")
    print("  - 在 C++ 里把当前步的「原始量」打日志或发话题：")
    print("      ang_vel(3), imu_roll_pitch(2), joint_pos_ros(12), joint_vel_ros(12), last_action_ros(12), foot_force_ros(4), default_dof_pos_ros(12)")
    print("  - 用本脚本里的 build_observation_cpp_style() 传入上述 raw 与 cpp_cfg，得到「期望的 53 维」")
    print("  - 若 C++ 同时发布当前步的 proprio（53 维），即可在 Python 里逐维对比，快速定位哪一维不一致。")
    print("  - 本脚本目录下可增加一个 ROS2 订阅节点，订阅上述原始量话题，并调用 build_observation_cpp_style 发布 /ref_proprio，便于与 /our_depth_rl/observation 等对比。")

    header("检查结束")
    print("若 2/3/4 均通过，且实际关节/足端接口顺序与配置中 dof_reindex、feet_reindex 一致，则观测与强化学习端对齐。\n")

if __name__ == "__main__":
    main()
