#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持续监测 C++ 发布的 proprio 观测（53 维）
- 计算每维的运行均值/方差（Welford）
- 定时输出整体统计与分段统计
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import argparse
import math

# 与训练端 obs_buf 前 53 维一致的分段
PROPRIO_SPEC = [
    (0,   3,  "base_ang_vel"),
    (3,   5,  "imu_rp"),
    (5,   8,  "yaw_info"),
    (8,  11,  "commands"),
    (11, 13,  "env_class"),
    (13, 25,  "dof_pos"),
    (25, 37,  "dof_vel"),
    (37, 49,  "last_actions"),
    (49, 53,  "contact"),
]

class RunningStats:
    def __init__(self, dim):
        self.dim = dim
        self.n = 0
        self.mean = [0.0] * dim
        self.m2 = [0.0] * dim
        self.min = [float("inf")] * dim
        self.max = [float("-inf")] * dim

    def update(self, x):
        self.n += 1
        for i, v in enumerate(x):
            # min/max
            if v < self.min[i]:
                self.min[i] = v
            if v > self.max[i]:
                self.max[i] = v
            # Welford
            delta = v - self.mean[i]
            self.mean[i] += delta / self.n
            delta2 = v - self.mean[i]
            self.m2[i] += delta * delta2

    def std(self):
        if self.n < 2:
            return [0.0] * self.dim
        return [math.sqrt(m / (self.n - 1)) for m in self.m2]

class ObsMonitor(Node):
    def __init__(self, topic, interval_sec, full_print):
        super().__init__("obs_monitor")
        self.topic = topic
        self.interval_sec = interval_sec
        self.full_print = full_print
        self.sub = self.create_subscription(Float32MultiArray, topic, self.cb, 10)
        self.stats = None
        self.last_print = time.time()
        self.msg_count = 0

    def cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        if not data:
            return
        if self.stats is None:
            self.stats = RunningStats(len(data))
            self.get_logger().info(f"init stats dim={len(data)} topic={self.topic}")
        if len(data) != self.stats.dim:
            self.get_logger().warn(f"dim mismatch: got {len(data)}, expected {self.stats.dim}")
            return
        self.stats.update(data)
        self.msg_count += 1

        now = time.time()
        if now - self.last_print >= self.interval_sec:
            self.last_print = now
            self.print_stats()

    def print_stats(self):
        mean = self.stats.mean
        std = self.stats.std()
        vmin = self.stats.min
        vmax = self.stats.max
        dim = self.stats.dim
        # overall summary
        overall_mean = sum(mean) / dim
        overall_std = sum(std) / dim
        overall_min = min(vmin)
        overall_max = max(vmax)
        self.get_logger().info(
            f"[{self.msg_count} msgs] overall mean={overall_mean:.4f} std={overall_std:.4f} "
            f"min={overall_min:.4f} max={overall_max:.4f}"
        )
        # block summary
        for s, e, name in PROPRIO_SPEC:
            if e > dim:
                continue
            blk_mean = sum(mean[s:e]) / (e - s)
            blk_std = sum(std[s:e]) / (e - s)
            blk_min = min(vmin[s:e])
            blk_max = max(vmax[s:e])
            self.get_logger().info(
                f"  {name:12s} mean={blk_mean:.4f} std={blk_std:.4f} min={blk_min:.4f} max={blk_max:.4f}"
            )
        if self.full_print:
            for i in range(dim):
                self.get_logger().info(
                    f"    idx[{i:02d}] mean={mean[i]:.4f} std={std[i]:.4f} min={vmin[i]:.4f} max={vmax[i]:.4f}"
                )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/our_depth_rl/proprio")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = ObsMonitor(args.topic, args.interval, args.full)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
