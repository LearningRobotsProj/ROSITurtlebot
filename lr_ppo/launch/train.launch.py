# lr_ppo/launch/train.launch.py
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
    
    # maze simulation
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
            'use_gui': use_gui
        }.items()
    )
    
    # PPO Training nodes - start with delays
    observation_vector = Node(
        package='lr_ppo',
        executable='observation_vector',
        name='observation_vector',
        output='screen'
    )
    
    reward_function = Node(
        package='lr_ppo',
        executable='reward_function',
        name='reward_function',
        output='screen'
    )
    
    collision_checker = Node(
        package='lr_ppo',
        executable='collision_checker',
        name='collision_checker',
        output='screen'
    )
    
    # Simple tester to verify robot moves
    movement_tester = Node(
        package='lr_ppo',
        executable='movement_tester',
        name='movement_tester',
        output='screen'
    )
    
    # PPO trainer node
    ppo_trainer = Node(
        package='lr_ppo',
        executable='ppo_trainer',
        name='ppo_trainer',
        output='screen',
        parameters=[{
            'total_episodes': 100,
            'max_steps_per_episode': 200,
            'model_save_path': './models/',
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 0.9,
            'epsilon_end': 0.1
        }]
    )
    
    # Create launch description with proper sequencing
    return LaunchDescription([
        # Start Gazebo simulation first
        maze_simulation,
        
        # Wait 8 seconds for Gazebo to fully initialize
        TimerAction(
            period=8.0,
            actions=[
                observation_vector,
            ]
        ),
        
        # Wait 2 more seconds for observation node
        TimerAction(
            period=10.0,
            actions=[
                reward_function,
                collision_checker,
            ]
        ),
        
        # Wait 2 more seconds, then test movement
        TimerAction(
            period=12.0,
            actions=[
                movement_tester,
            ]
        ),
        
        # Finally start training after movement test completes
        TimerAction(
            period=30.0,  # Give time for movement test
            actions=[
                ppo_trainer,
            ]
        ),
    ])