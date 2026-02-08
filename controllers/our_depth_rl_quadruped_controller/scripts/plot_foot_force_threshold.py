#!/usr/bin/env python3
"""
实时绘制四足足端力与接触力阈值，用于调试 foot_force_threshold。
订阅 /our_depth_rl/foot_force_debug（Float32MultiArray，5 个元素：[FL, FR, RL, RR, threshold]），
绘制 4 条力曲线 + 阈值水平线；力高于阈值判为接触。
左上角显示：收包速率(Hz)、最近一帧时间，便于判断是否实时。
"""
import collections
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import matplotlib.pyplot as plt
import numpy as np


class FootForceThresholdPlot(Node):
    def __init__(self):
        super().__init__("foot_force_threshold_plot")
        self.declare_parameter("topic", "/our_depth_rl/foot_force_debug")
        self.declare_parameter("window_sec", 3.0)
        self.declare_parameter("hz", 50.0)
        self.declare_parameter("legs", ["FL", "FR", "RL", "RR"])

        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.window_sec = self.get_parameter("window_sec").get_parameter_value().double_value
        self.hz = self.get_parameter("hz").get_parameter_value().double_value
        self.legs = list(self.get_parameter("legs").get_parameter_value().string_array_value)

        self.max_len = max(50, int(self.window_sec * self.hz))
        self.buf = collections.deque(maxlen=self.max_len)

        self.msg_count = 0
        self.last_rate_time = time.time()
        self.msg_rate = 0.0
        self.last_msg_time = 0.0

        self.sub = self.create_subscription(Float32MultiArray, self.topic, self.cb, 10)

        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        self.lines = {}
        self.leg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, leg in enumerate(self.legs):
            (ln,) = self.ax.plot([], [], label=leg, color=self.leg_colors[i], lw=1.5)
            self.lines[leg] = ln
        (self.ln_thr,) = self.ax.plot([], [], "k--", label="threshold", lw=1.2)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Force [N]")
        self.ax.set_title("Foot force vs threshold (above line = contact)")
        self.ax.legend(loc="upper right")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 80)
        self.ax.set_xlim(0, self.window_sec)
        self.status_text = self.fig.text(0.02, 0.98, "", fontsize=10, verticalalignment="top",
                                         family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])

        self.t0 = time.time()
        self.timer = self.create_timer(1.0 / self.hz, self.update_plot)

    def cb(self, msg: Float32MultiArray):
        if len(msg.data) < 5:
            return
        now = time.time()
        self.last_msg_time = now
        self.msg_count += 1
        if now - self.last_rate_time >= 0.5:
            self.msg_rate = self.msg_count / (now - self.last_rate_time)
            self.msg_count = 0
            self.last_rate_time = now
        t = now - self.t0
        self.buf.append([t, msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4]])

    def update_plot(self):
        now = time.time()
        if not self.buf:
            self.status_text.set_text(f"Waiting for {self.topic}\n(ROS_DOMAIN_ID same as controller?)")
            self.ax.set_xlim(0, self.window_sec)
            self.ax.set_ylim(0, 80)
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            return
        arr = np.array(self.buf)
        t = arr[:, 0]
        thr = arr[:, 5]
        for i, leg in enumerate(self.legs):
            self.lines[leg].set_data(t, arr[:, 1 + i])
        self.ln_thr.set_data(t, thr)
        t_now = t[-1]
        self.ax.set_xlim(max(0, t_now - self.window_sec), t_now)
        ymax = max(np.nanmax(arr[:, 1:5]) * 1.15, np.nanmax(thr) * 1.2, 10.0)
        self.ax.set_ylim(0, max(ymax, 50))
        age = now - self.last_msg_time
        self.status_text.set_text(f"Rate: {self.msg_rate:.1f} Hz\nLast: {age*1000:.0f} ms ago")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = FootForceThresholdPlot()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
