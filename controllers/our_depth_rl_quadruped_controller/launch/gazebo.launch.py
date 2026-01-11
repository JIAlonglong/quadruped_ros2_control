import os

import xacro
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, 
    OpaqueFunction, 
    IncludeLaunchDescription, 
    RegisterEventHandler
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(context, *args, **kwargs):
    # 增加日志级别和性能监控参数
    os.environ['RCUTILS_LOGGING_BUFFERED_STREAM'] = '1'
    os.environ['RCUTILS_LOGGING_USE_STDOUT'] = '0'
    os.environ['RCUTILS_COLORIZED_OUTPUT'] = '1'
    
    package_description = context.launch_configurations['pkg_description']
    init_height = context.launch_configurations['height']
    pkg_path = os.path.join(
        get_package_share_directory(package_description)
    )

    xacro_file = os.path.join(
        pkg_path, 'xacro', 'robot.xacro'
    )
    robot_description = xacro.process_file(xacro_file, mappings={'GAZEBO': 'true'}).toxml()

    rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", "visualize_urdf.rviz")

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_ocs2',
        output='screen',
        arguments=["-d", rviz_config_file],
        parameters=[
            {'use_sim_time': True}
        ]
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name',
                   'robot', '-allow_renaming', 'true', '-z', init_height],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'publish_frequency': 50.0,  # 增加发布频率
                'use_tf_static': True,
                'robot_description': robot_description,
                'ignore_timestamp': True,
                'use_sim_time': True  # 添加use_sim_time参数
            }
        ],
        output='screen'
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

    rl_quadruped_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rl_quadruped_controller", "--controller-manager", "/controller_manager"],
    )

    # 创建时钟和图像桥接节点
    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/rgbd_d435/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked'
        ],
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ],
        respawn=True,
        respawn_delay=2.0
    )

    # RGB图像桥接节点
    gz_image_bridge_node = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=[
            '/camera/image',
            '/rgbd_d435/image',
            '/rgbd_d435/depth_image'
        ],
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'camera.image.compressed.jpeg_quality': 75
            }
        ]
    )
    
    return [
        # 先启动时钟和图像桥接
        gz_bridge_node,
        gz_image_bridge_node,
        
        # 启动Gazebo仿真，降低日志级别提高性能
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                       'launch',
                                       'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', [' -r -v 3 empty.sdf'])]  # 降低日志级别
        ),
        
        # 等待时钟桥接启动后，再启动机器人状态发布器
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_bridge_node,
                on_exit=[robot_state_publisher]
            )
        ),
        
        # 等待状态发布器启动后，再启动RViz和生成机器人实体
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=robot_state_publisher,
                on_exit=[rviz, gz_spawn_entity]
            )
        ),
        
        # 等待机器人实体生成后，再启动广播器
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[
                    imu_sensor_broadcaster, 
                    joint_state_publisher
                ],
            )
        ),
        
        # 最后启动RL控制器
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[rl_quadruped_controller],
            )
        ),
    ]


def generate_launch_description():
    pkg_description = DeclareLaunchArgument(
        'pkg_description',
        default_value='go2_description',
        description='package for robot description'
    )

    height = DeclareLaunchArgument(
        'height',
        default_value='0.5',
        description='Init height in simulation'
    )

    return LaunchDescription([
        pkg_description,
        height,
        OpaqueFunction(function=launch_setup),
    ])
