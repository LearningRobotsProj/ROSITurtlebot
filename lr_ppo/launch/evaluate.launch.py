# lr_ppo/launch/evaluate.launch.py
#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    maze_arg = 'maze1'  
    
    #  maze simulation
    maze_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('lr_turtlebot_sim'),
                'launch',
                'maze_world.launch.py'
            ])
        ]),
        launch_arguments={
            'world': maze_arg
        }.items()
    )
    
    # Evaluation nodes
    observation_vector = Node(
        package='lr_ppo',
        executable='observation_vector',
        name='observation_vector',
        output='screen'
    )
    
    collision_checker = Node(
        package='lr_ppo',
        executable='collision_checker',
        name='collision_checker',
        output='screen'
    )
    
    rosi_evaluator = Node(
        package='lr_ppo',
        executable='rosi_evaluator',
        name='rosi_evaluator',
        output='screen',
        parameters=[{
            'num_evaluation_episodes': 20,
            'maze_name': maze_arg,
            'model_path': './models/rosi_ppo_final.pth'
        }]
    )
    
    return LaunchDescription([
        maze_simulation,
        observation_vector,
        collision_checker,
        rosi_evaluator
    ])