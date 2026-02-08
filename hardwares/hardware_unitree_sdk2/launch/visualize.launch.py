import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    package_description = context.launch_configurations['pkg_description']
    pkg_path = os.path.join(get_package_share_directory(package_description))

    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare(package_description),
            "config",
            "robot_control.yaml",
        ]
    )

    # depth_go2 时加载带深度/图像面板的配置，便于查看感知图像
    rviz_config = "visualize_images.rviz" if package_description == "depth_go2_description" else "visualize_urdf.rviz"
    rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", rviz_config)

    # 将 rviz2 日志设为 WARN，避免 "Message Filter dropping message (queue full)" 等 INFO 刷屏
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_ocs2',
        output='screen',
        arguments=["-d", rviz_config_file, "--ros-args", "--log-level", "WARN"]
    )

    # 降低日志级别，减少刷屏；保留控制器侧启动与推理信息
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                'publish_frequency': 20.0,
                'use_tf_static': True,
                'robot_description': robot_description,
                'ignore_timestamp': True
            }
        ],
    )

    # 控制器进程：保留 INFO 以显示启动信息与 [推理] 延迟；其余冗余日志已改为 DEBUG
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_controllers],
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
        output="both",
    )

    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    our_depth_rl_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "our_depth_rl_quadruped_controller",
            "--controller-manager",
            "/controller_manager",
            "-t",
            "our_depth_rl_quadruped_controller/LeggedGymController",
            "--param-file",
            robot_controllers,
        ],
    )

    # 降低分辨率/帧率可减少 bad_alloc（尤其 WSL2/低内存环境），控制器内部会 crop/resize
    realsense_camera = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="rgbd_d435",
        name="camera",
        output="screen",
        parameters=[
            {
                "enable_color": True,
                "enable_depth": True,
                "depth_module.depth_profile": "640x480x15",
                "rgb_camera.color_profile": "640x480x15",
                "pointcloud.enable": False,
            }
        ],
        remappings=[
            ("color/image_raw", "image"),
            ("depth/image_rect_raw", "depth_image"),
        ],
    )

    return [
        rviz,
        robot_state_publisher,
        # realsense_camera,  # 不在此 launch 启动相机；需图像时单独起 RealSense 或仿真
        controller_manager,
        joint_state_publisher,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[imu_sensor_broadcaster, our_depth_rl_controller],
            )
        )
    ]


def generate_launch_description():
    pkg_description = DeclareLaunchArgument(
        'pkg_description',
        default_value='go2_description',
        description='package for robot description'
    )
    ros_domain_id_arg = DeclareLaunchArgument(
        'ros_domain_id',
        default_value='1',
        description='ROS 2 DDS domain ID，需与 ros2_control.xacro 中 Unitree SDK 的 domain 一致，避免 PreconditionNotMetError'
    )

    # 整次启动使用 domain 1，与 ros2_control.xacro 中 Unitree SDK 一致，避免 DDS 冲突
    return LaunchDescription([
        SetEnvironmentVariable('ROS_DOMAIN_ID', '1'),
        pkg_description,
        ros_domain_id_arg,
        OpaqueFunction(function=launch_setup),
    ])
