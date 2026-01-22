#!/usr/bin/env python3
import collections
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import matplotlib.pyplot as plt
import numpy as np


class ContactGaitPlot(Node):
    def __init__(self):
        super().__init__("contact_gait_plot")
        self.declare_parameter("topic", "/our_depth_rl/contact_states")
        self.declare_parameter("window_sec", 2.0)
        self.declare_parameter("hz", 50.0)
        # 显示顺序：真实足端顺序（ROS）
        self.declare_parameter("legs", ["FL", "FR", "RL", "RR"])
        # feet_reindex: ROS 顺序 -> 策略顺序（用于还原显示）
        self.declare_parameter("feet_reindex", [1, 0, 3, 2])

        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.window_sec = self.get_parameter("window_sec").get_parameter_value().double_value
        self.hz = self.get_parameter("hz").get_parameter_value().double_value
        self.legs = list(self.get_parameter("legs").get_parameter_value().string_array_value)
        self.feet_reindex = list(self.get_parameter("feet_reindex").get_parameter_value().integer_array_value)
        self.feet_reindex_inv = [0, 1, 2, 3]
        if len(self.feet_reindex) == 4:
            for sim_idx, ros_idx in enumerate(self.feet_reindex):
                if 0 <= ros_idx < 4:
                    self.feet_reindex_inv[sim_idx] = ros_idx

        self.max_len = max(10, int(self.window_sec * self.hz))
        self.buf = collections.deque([np.zeros(4, dtype=np.float32)] * self.max_len, maxlen=self.max_len)

        self.sub = self.create_subscription(Float32MultiArray, self.topic, self.cb, 10)

        self.fig, self.ax = plt.subplots(figsize=(6, 2.2))
        self.im = self.ax.imshow(
            np.zeros((4, self.max_len), dtype=np.float32),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        self.ax.set_yticks(range(4))
        self.ax.set_yticklabels(self.legs)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_title("Foot Contact (1=contact)")
        self.fig.tight_layout()

        self.timer = self.create_timer(1.0 / self.hz, self.update_plot)
        self.last_ts = time.time()

    def cb(self, msg: Float32MultiArray):
        if len(msg.data) < 4:
            return
        # contact 值为 {-0.5, +0.5}，映射到 {0,1}
        contact = np.array(msg.data[:4], dtype=np.float32)
        contact = (contact > 0.0).astype(np.float32)
        # 输入为策略顺序，按 feet_reindex_inv 还原到 ROS 顺序显示
        contact_ros = contact[self.feet_reindex_inv]
        self.buf.append(contact_ros)

    def update_plot(self):
        data = np.stack(self.buf, axis=1)  # [4, T]
        self.im.set_data(data)
        # x 轴以时间显示
        now = time.time()
        if now - self.last_ts > 0.5:
            self.ax.set_xticks([0, self.max_len // 2, self.max_len - 1])
            self.ax.set_xticklabels([f"{self.window_sec:.1f}", f"{self.window_sec/2:.1f}", "0"])
            self.last_ts = now
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = ContactGaitPlot()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

