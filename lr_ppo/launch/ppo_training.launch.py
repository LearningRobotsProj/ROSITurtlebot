#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
import os

def generate_launch_description():
    # Check if we're in a container or have display
    use_gui = LaunchConfiguration('use_gui', default='true')
    
    # Include the maze simulation launch file
    maze_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('lr_turtlebot_sim'),
                'launch',
                'maze_world.launch.py'
            ])
        ]),
        launch_arguments={
            'world': 'maze1',
            'use_gui': use_gui,
            'use_rviz': 'false',  # We'll use Gazebo GUI instead
        }.items()
    )
    
    # Start a bridge for ROS 1/ROS 2 communication (if needed)
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        arguments=[
            # Clock bridge (critical)
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            
            # TF bridges
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            
            # IMU bridge
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            
            # Camera bridges
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            
            # Laser scan bridge
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            
            # Odom bridge
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ],
        output='screen'
    )
    
    # Observation vector node
    observation_vector = Node(
        package='lr_ppo',
        executable='observation_vector',
        name='observation_vector',
        output='screen',
        parameters=[{
            'normalize': True,
            'laser_num_samples': 20,
            'goal_x': 5.0,
            'goal_y': 5.0,
        }]
    )
    
    # Reward function node
    reward_function = Node(
        package='lr_ppo',
        executable='reward_function',
        name='reward_function',
        output='screen',
        parameters=[{
            'reward_goal_reached': 100.0,
            'penalty_collision': -50.0,
            'reward_distance_multiplier': 5.0,
            'penalty_time_step': -0.01,
            'max_steps': 500,
            'goal_threshold': 0.5,
            'angle_reward_multiplier': 0.5,
            'velocity_bonus': 0.02,
        }]
    )
    
    # Collision checker node
    collision_checker = Node(
        package='lr_ppo',
        executable='collision_checker',
        name='collision_checker',
        output='screen',
        parameters=[{
            'collision_threshold': 0.15,
            'goal_threshold': 0.5,
            'timeout_steps': 500,
            'front_collision_threshold': 0.18,
            'side_collision_threshold': 0.15,
        }]
    )
    
    # PPO trainer node (using YOUR custom PPO agent)
    ppo_trainer = Node(
        package='lr_ppo',
        executable='ppo_trainer',
        name='ppo_trainer',
        output='screen',
        parameters=[{
            'max_episodes': 1000,
            'max_steps_per_episode': 500,
            'update_frequency': 512,
            'save_frequency': 5000,
            'model_dir': './models/ppo_turtlebot3',
            'log_dir': './training_logs',
        }]
    )
    
    # Optional: RViz for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('lr_turtlebot_sim'),
            'config',
            'turtlebot3_navigation.rviz'
        ])],
        condition=IfCondition(use_gui),
        output='screen'
    )
    
    # Launch description with proper sequencing
    return LaunchDescription([
        # Start Gazebo simulation FIRST (most important)
        maze_simulation,
        
        # Wait for Gazebo to initialize, then start bridge
        TimerAction(
            period=5.0,
            actions=[parameter_bridge]
        ),
        
        # Wait for bridge to connect, then start observation node
        TimerAction(
            period=10.0,
            actions=[observation_vector]
        ),
        
        # Wait a bit more, then start reward and collision nodes
        TimerAction(
            period=12.0,
            actions=[
                reward_function,
                collision_checker,
            ]
        ),
        
        # Optional: Start RViz for visualization
        TimerAction(
            period=15.0,
            actions=[rviz_node],
            condition=IfCondition(use_gui)
        ),
        
        # Finally start PPO training after everything is ready
        TimerAction(
            period=18.0,  # Give time for all nodes to initialize
            actions=[ppo_trainer]
        ),
    ])