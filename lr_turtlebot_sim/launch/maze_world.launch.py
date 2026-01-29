#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    maze_arg = DeclareLaunchArgument(
        'world',
        default_value='maze1',
        description='Maze world to load'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true'
    )

    sim_pkg_share = FindPackageShare('lr_turtlebot_sim')
    ros_gz_sim_share = FindPackageShare('ros_gz_sim')

    world_file = PathJoinSubstitution([
        sim_pkg_share,
        'worlds',
        [LaunchConfiguration('world'), TextSubstitution(text='.world')]
    ])

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([ros_gz_sim_share, 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={
            'gz_args': [world_file, TextSubstitution(text=' -r -v 4')],
        }.items()
    )

    spawn_robot = Node(
        package='lr_turtlebot_sim',
        executable='spawn_robot',
        output='screen'
    )

    maze_simulator = Node(
        package='lr_turtlebot_sim',
        executable='maze_simulator',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_name': 'turtlebot3_burger'
        }]
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'
        ],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return LaunchDescription([
        maze_arg,
        use_sim_time_arg,
        gz_sim_launch,

        TimerAction(
            period=2.0,
            actions=[
                LogInfo(msg='Gazebo started, spawning robot'),
                spawn_robot
            ]
        ),

        TimerAction(
            period=5.0,
            actions=[
                bridge,
                maze_simulator
            ]
        )
    ])
