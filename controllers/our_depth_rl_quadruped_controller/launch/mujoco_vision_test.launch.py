"""
MuJoCo vision test (unitree_mujoco + ros2_control + our_depth_rl_quadruped_controller)

Usage (2 terminals recommended):

Terminal A (MuJoCo simulator with ROS2 image publishing):
  export MUJOCO_GL=egl   # if headless; otherwise omit
  export ROS_DOMAIN_ID=1
  python3 /home/jialonglong/Quadruped_ws/src/quadruped_ros2_control/hardwares/unitree_mujoco/simulate_python/unitree_mujoco.py

Terminal B (controller):
  source /home/jialonglong/Quadruped_ws/install/setup.bash
  export ROS_DOMAIN_ID=1
  ros2 launch our_depth_rl_quadruped_controller mujoco.launch.py use_rviz:=true

This launch file is intentionally lightweight and just delegates to mujoco.launch.py.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("our_depth_rl_quadruped_controller"),
                            "launch",
                            "mujoco.launch.py",
                        ]
                    )
                ),
            )
        ]
    )


