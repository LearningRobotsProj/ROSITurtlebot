# lr_ppo/launch/train.launch.py
#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Include maze simulation
    maze_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('lr_turtlebot_sim'),
                'launch',
                'maze_world.launch.py'
            ])
        ]),
        launch_arguments={
            'world': 'maze1'
        }.items()
    )
    
    # PPO Training nodes
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
    
    ppo_trainer = Node(
        package='lr_ppo',
        executable='ppo_trainer',
        name='ppo_trainer',
        output='screen',
        parameters=[{
            'total_episodes': 1000,  # Reduced for testing
            'max_steps_per_episode': 500,
            'model_save_path': './models/'
        }]
    )
    
    return LaunchDescription([
        maze_simulation,
        observation_vector,
        reward_function,
        collision_checker,
        ppo_trainer
    ])