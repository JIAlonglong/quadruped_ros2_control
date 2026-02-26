import math
import os
import random

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


package_description = "depth_go2_description"


def process_xacro(camera_pitch: float = 0.0):
    # 生成 robot_description（URDF XML 字符串），供：
    # - robot_state_publisher 发布 TF
    # - ros_gz_sim/create 在 Gazebo 中生成实体
    pkg_path = os.path.join(get_package_share_directory(package_description))
    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    robot_description_config = xacro.process_file(
        xacro_file,
        mappings={
            'GAZEBO': 'true',
            # 在 xacro 中打开外部传感器模块（例如 RGBD 相机 / 深度图 / 点云）
            'EXTERNAL_SENSORS': 'true',
            'CAMERA_PITCH': str(camera_pitch),
        }
    )
    return robot_description_config.toxml()


def generate_launch_description():
    # RViz 配置：包含图像/点云等显示面板，便于观察深度与 RGB 数据是否正常
    rviz_config_file = os.path.join(
        get_package_share_directory(package_description), 
        "config", 
        "visualize_images.rviz"  # 使用添加了相机和点云显示的配置文件
    )

    # ros2_control 控制器配置（controller_manager 使用）
    controller_config_file = os.path.join(
        get_package_share_directory(package_description), 
        "config", 
        "robot_control.yaml"
    )

    # 地形切换：通过 launch 参数 world 选择 worlds/ 下的 .sdf 文件
    # 例如：ros2 launch ... gazebo_rl_control.launch.py world:=room_with_obstacles.sdf
    # 可选：parkour_with_sensors.sdf、empty_with_sensors.sdf、room_with_obstacles.sdf（房间+棕/绿高台+斜坡）
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="parkour_with_sensors.sdf",
        description="World file in worlds/ (e.g. parkour_with_sensors.sdf, empty_with_sensors.sdf)",
    )
    world_file = PathJoinSubstitution(
        [FindPackageShare(package_description), "worlds", LaunchConfiguration("world")]
    )
    camera_pitch_rad = 0.30
    robot_description = process_xacro(camera_pitch_rad)

    # 通过 ros_gz_sim 的 create 可执行程序，把 robot_description 里的模型生成到 Gazebo
    # 机器人放在地面、距离高台 2m（高台 1 中心在 x=0，故 spawn 在 x=-2）
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'robot',
            '-allow_renaming', 'true',
            '-x', '-2.0',
            '-y', '0.0',
            '-z', '0.4',
        ],
    )

    # 发布 TF：把 URDF 的 joint/link 关系发布到 /tf 与 /tf_static
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'publish_frequency': 20.0,
                'use_tf_static': True,
                'robot_description': robot_description,
                'ignore_timestamp': True
            }
        ],
    )

    # 启动 joint_state_broadcaster：发布 /joint_states（以及 controller_manager 的状态）
    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager"
        ],
    )

    # 启动 IMU broadcaster：把仿真 IMU 数据发布为 sensor_msgs/Imu
    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager", "/controller_manager"
        ],
    )

    # 启动 RL 控制器（depth_rl_quadruped_controller）
    # - spawner 会请求 controller_manager 加载并激活对应 controller
    # - --param-file 传入 robot_control.yaml（包含 joints/接口类型/站立姿态等）
    controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "our_depth_rl_quadruped_controller", 
            "--controller-manager", "/controller_manager",
            "-t", "our_depth_rl_quadruped_controller/LeggedGymController",
            "--param-file", controller_config_file
        ],
        parameters=[
            {
                'config_folder': os.path.join(
                    get_package_share_directory(package_description), 
                    'config',
                    'extreme_parkour_common'
                ),
                # 启用独立RL线程以匹配控制器实现
                'use_rl_thread': True,
                # 让控制器加载深度相机策略
                'use_camera': True,
                # 再次显式声明模型包与配置目录，便于策略寻找TorchScript文件
                'robot_pkg': package_description,
                'model_folder': 'extreme_parkour_common',
            }
        ],
    )

    # Gazebo <-> ROS2 桥接：
    # - /clock：仿真时钟
    # - /rgbd_d435/*：相机信息、点云、深度图、RGB 图
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/rgbd_d435/camera_info@sensor_msgs/msg/CameraInfo' +
            '[gz.msgs.CameraInfo',
            '/rgbd_d435/points@sensor_msgs/msg/PointCloud2' +
            '[gz.msgs.PointCloudPacked',
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
        )

    # 用 ros_gz_image 处理图像桥接：参数为 (gz 话题, ros 话题)；若深度 100% 无效，请先启动仿真后
    # 运行 gz topic -l | grep -E "image|depth" 查看 Gazebo 实际话题名，若带 /world/.../model/robot/... 前缀，
    # 需把此处第一个参数改为实际 gz 深度话题（或确保 D435 模型 <topic>/rgbd_d435</topic> 在 gz 中生效）。
    gz_image_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=[
            '/rgbd_d435/image',
            '/rgbd_d435/depth_image',
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ],
    )

    # 延迟启动各个 controller spawner：
    # controller_manager / GazeboSystem 有时会在启动早期尚未 ready，直接 spawner 容易失败
    delayed_controller_spawners = TimerAction(
        period=5.0,
        actions=[
            joint_state_publisher,
            imu_sensor_broadcaster,
            controller,
        ]
    )

    return LaunchDescription([
        SetEnvironmentVariable(name="DEPTH_INPUT_SOURCE", value="ros"),
        world_arg,
        # RViz：用仿真时钟，避免 TF/图像时间戳不一致导致显示异常
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz_ocs2',
            output='screen',
            arguments=['-d', rviz_config_file],
            parameters=[
                {'use_sim_time': True}  # 启用仿真时间，与Gazebo同步
            ]
        ),

        # 桥接节点建议尽早启动，方便后续节点立即拿到 /clock 与相机话题
        gz_bridge_node,
        gz_image_bridge_node,

        # 启动 Gazebo Sim（通过 include 官方 gz_sim.launch.py）
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    PathJoinSubstitution([
                        FindPackageShare('ros_gz_sim'),
                        'launch',
                        'gz_sim.launch.py'
                    ])
                ]
            ),
            launch_arguments=[('gz_args', ['-r -v 4 ', world_file])]
        ),

        # 在 Gazebo 中生成机器人实体
        gz_spawn_entity,

        # 发布机器人 TF
        robot_state_publisher,

        # 延迟启动各个控制器 spawner
        delayed_controller_spawners,
    ])
