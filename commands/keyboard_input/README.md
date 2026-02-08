# Keyboard Input

This node will read the keyboard input and publish a control_input_msgs/Input message.

Tested environment:
* Ubuntu 24.04
  * ROS2 Jazzy
* Ubuntu 22.04
  * ROS2 Humble

### Build Command
```bash
cd ~/ros2_ws
colcon build --packages-up-to keyboard_input
```

### Launch Command
```bash
source ~/ros2_ws/install/setup.bash
ros2 run keyboard_input keyboard_input
```

## 1. Use Instructions for Unitree Guide
### 1.1 Control Mode
* Passive Mode: Keyboard 1
* Fixed Stand: Keyboard 2
    * Free Stand: Keyboard 3
    * Trot: Keyboard 4
    * SwingTest: Keyboard 5
    * Balance: Keyboard 6
### 1.2 Control Input
* WASD IJKL: Move robot
* Space: Reset Speed Input

## 2. our_depth_rl 控制器（Extreme Parkour / direction_mode）
* **必须先进入 RL 状态，键盘方向才生效**：启动后默认是 PASSIVE，按 **3** 进入 RL 后，J/L 才控制 yaw。
* 按键 **3**：PASSIVE → RL（站立策略控制）；在 RL 下 J=左转、L=右转。
* 按键 **2**：PASSIVE → 趴下；在 FIXEDSTAND 下按 2 可趴下，再按 2 站立，再按 3 进入 RL。
* 请在本机**有焦点的终端**里运行 `ros2 run keyboard_input keyboard_input`，否则收不到按键；launch 不会自动启动键盘节点。