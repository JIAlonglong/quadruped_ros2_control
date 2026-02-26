import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # NOTE:
    # - `robot_description_pkg`: which URDF/xacro to use (for MuJoCo it should use hardware_unitree_mujoco)
    # - `controller_config_pkg`: where robot_control.yaml + rviz configs live (for our depth RL it's depth_go2_description)
    robot_description_pkg = context.launch_configurations["robot_description_pkg"]
    controller_config_pkg = context.launch_configurations["controller_config_pkg"]
    use_rviz = context.launch_configurations["use_rviz"]
    start_realsense = context.launch_configurations["start_realsense"]
    start_unitree_mujoco_sim = context.launch_configurations["start_unitree_mujoco_sim"]

    robot_pkg_path = os.path.join(get_package_share_directory(robot_description_pkg))
    ctrl_pkg_path = os.path.join(get_package_share_directory(controller_config_pkg))

    xacro_file = os.path.join(robot_pkg_path, "xacro", "robot.xacro")
    robot_description = xacro.process_file(xacro_file, mappings={"MUJOCO": "true"}).toxml()

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare(controller_config_pkg),
            "config",
            "robot_control.yaml",
        ]
    )

    # MuJoCo 视觉策略默认需要看相机流，直接用带图像面板的 RViz 配置。
    rviz_config_file = os.path.join(ctrl_pkg_path, "config", "visualize_images.rviz")

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_ocs2",
        output="screen",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(use_rviz),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            {
                "publish_frequency": 20.0,
                "use_tf_static": True,
                "robot_description": robot_description,
                "ignore_timestamp": True,
            }
        ],
    )

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
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    # Our depth RL controller
    controller = Node(
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

    # Optional: use a real D435 in MuJoCo test (publish /rgbd_d435/{image,depth_image})
    # If you don't have a physical camera, keep it false and provide images via rosbag / your own publisher.
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
            }
        ],
        remappings=[
            ("color/image_raw", "image"),
            ("depth/image_rect_raw", "depth_image"),
        ],
        condition=IfCondition(start_realsense),
    )

    # Optional: start Unitree MuJoCo C++ simulator (in this repo) as a subprocess.
    # This is useful to ensure the full chain is started consistently:
    #   unitree_mujoco (DDS)  <->  hardware_unitree_mujoco (ros2_control)  <->  our controller (ROS2)
    mujoco_gl = LaunchConfiguration("mujoco_gl")
    unitree_mujoco_sim_bin = LaunchConfiguration("unitree_mujoco_sim_bin")
    unitree_mujoco_robot = LaunchConfiguration("unitree_mujoco_robot")
    unitree_mujoco_scene = LaunchConfiguration("unitree_mujoco_scene")
    unitree_mujoco_domain = LaunchConfiguration("unitree_mujoco_domain")
    unitree_mujoco_iface = LaunchConfiguration("unitree_mujoco_iface")
    cyclonedds_uri = LaunchConfiguration("cyclonedds_uri")
    force_cyclonedds_uri = LaunchConfiguration("force_cyclonedds_uri")

    unitree_mujoco_sim = ExecuteProcess(
        cmd=[
            unitree_mujoco_sim_bin,
            "-r",
            unitree_mujoco_robot,
            "-s",
            unitree_mujoco_scene,
            "-i",
            unitree_mujoco_domain,
            "-n",
            unitree_mujoco_iface,
        ],
        output="screen",
        condition=IfCondition(start_unitree_mujoco_sim),
    )

    # Only set MUJOCO_GL if user explicitly provided a non-empty value.
    # Empty (default) lets MuJoCo auto-detect the best backend (avoids EGL crash in WSLg).
    mujoco_gl_str = context.launch_configurations.get("mujoco_gl", "")
    env_actions = []
    if mujoco_gl_str:
        env_actions.append(SetEnvironmentVariable(name="MUJOCO_GL", value=mujoco_gl))
    # Critical: keep Unitree's CycloneDDS libs ahead of ROS CycloneDDS libs.
    # Mixed libddsc (ROS) + libddscxx (Unitree) causes runtime crash: free(): invalid pointer.
    env_actions.append(
        SetEnvironmentVariable(
            name="LD_LIBRARY_PATH",
            value=["/opt/unitree_robotics/lib:", EnvironmentVariable("LD_LIBRARY_PATH", default_value="")],
        )
    )

    return env_actions + [
        SetEnvironmentVariable(name="DEPTH_INPUT_SOURCE", value="dds"),
        # Keep ROS2 domain aligned with the simulator/domain used for bringup.
        SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=unitree_mujoco_domain),
        # Unitree SDK2 uses CycloneDDS internally (ddsc). Override any bad inherited config (e.g. eth2)
        # with a minimal loopback-only config unless explicitly disabled.
        SetEnvironmentVariable(name="CYCLONEDDS_URI", value=cyclonedds_uri, condition=IfCondition(force_cyclonedds_uri)),
        unitree_mujoco_sim,
        rviz,
        robot_state_publisher,
        controller_manager,
        realsense_camera,
        joint_state_publisher,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[imu_sensor_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=imu_sensor_broadcaster,
                on_exit=[controller],
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_description_pkg",
            default_value="depth_go2_description",
            description="URDF/xacro package used to generate robot_description (MuJoCo uses hardware_unitree_mujoco here).",
        ),
        DeclareLaunchArgument(
            "controller_config_pkg",
            default_value="depth_go2_description",
            description="Package that provides config/robot_control.yaml and rviz configs.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Whether to start RViz.",
        ),
        DeclareLaunchArgument(
            "start_realsense",
            default_value="false",
            description="Start realsense2_camera (requires a physical D435) to publish /rgbd_d435 images.",
        ),
        DeclareLaunchArgument(
            "start_unitree_mujoco_sim",
            default_value="false",
            description="Start the Unitree MuJoCo C++ simulator as a subprocess (recommended for one-click bringup).",
        ),
        DeclareLaunchArgument(
            "unitree_mujoco_sim_bin",
            default_value="/home/jialonglong/Quadruped_ws/build/unitree_mujoco/unitree_mujoco",
            description="Path to unitree_mujoco C++ simulator binary.",
        ),
        DeclareLaunchArgument(
            "unitree_mujoco_robot",
            default_value="go2",
            description="Robot type for unitree_mujoco (-r).",
        ),
        DeclareLaunchArgument(
            "unitree_mujoco_scene",
            default_value="scene.xml",
            description="Scene file name for unitree_mujoco (-s). Resolved relative to unitree_robots/<robot>/ if not an absolute path.",
        ),
        DeclareLaunchArgument(
            "unitree_mujoco_domain",
            default_value="1",
            description="DDS domain id for unitree_mujoco (-i). Must match hardware_unitree_mujoco domain.",
        ),
        DeclareLaunchArgument(
            "unitree_mujoco_iface",
            default_value="lo",
            description="DDS network interface for unitree_mujoco (-n). Must match hardware_unitree_mujoco network_interface.",
        ),
        DeclareLaunchArgument(
            "mujoco_gl",
            default_value="",
            description="MUJOCO_GL value. Empty=auto detect. Use 'egl' for headless, 'glfw' for WSLg desktop.",
        ),
        DeclareLaunchArgument(
            "cyclonedds_uri",
            default_value="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"lo\"/></Interfaces></General></Domain></CycloneDDS>",
            description="CycloneDDS config XML string for Unitree SDK2 (ddsc). Default: loopback only.",
        ),
        DeclareLaunchArgument(
            "force_cyclonedds_uri",
            default_value="false",
            description="If true, override CYCLONEDDS_URI env var. Default false: Unitree SDK2 manages DDS internally, external override may cause 'Failed to create domain' crash.",
        ),
        OpaqueFunction(function=launch_setup),
    ])
