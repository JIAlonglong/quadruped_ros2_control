#!/usr/bin/env python3

import os
from ament_index_python.packages import (
    get_package_share_directory
)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 获取包的共享目录路径
    package_description = 'depth_rl_quadruped_controller'
    pkg_path = os.path.join(
        get_package_share_directory(package_description)
    )

    # RViz配置文件路径
    rviz_config_file = os.path.join(
        pkg_path, "config", "image_visualizer.rviz")

    # 创建RViz节点
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='image_visualizer',
        output='screen',
        arguments=["-d", rviz_config_file],
        parameters=[
            {'use_sim_time': True}
        ]
    )

    # 创建并返回启动描述
    return LaunchDescription([
        rviz
    ])