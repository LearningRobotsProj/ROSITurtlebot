# lr_turtlebot_sim/launch/maze_world.launch.py
#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Launch arguments
    maze_arg = DeclareLaunchArgument(
        'world',
        default_value='maze1',
        description='Maze world to load: maze1, maze2, or maze3'
    )
    
    robot_arg = DeclareLaunchArgument(
        'model',
        default_value='burger',
        description='TurtleBot3 model: burger, waffle, or waffle_pi'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Get package directory
    pkg_dir = FindPackageShare('lr_turtlebot_sim')
    
    # Create world path - fixed by using a list for concatenation
    world_path = PathJoinSubstitution([
        pkg_dir,
        'worlds',
        LaunchConfiguration('world')
    ])
    
    # We'll add .world in the command using shell expansion
    start_gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose',
            # Use the substitution
            world_path,
            '.world',  # Add extension separately
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen'
    )
    
    # Create entity name - fixed by using list concatenation
    entity_name = ['turtlebot3_', LaunchConfiguration('model')]
    
    # Spawn TurtleBot3
    spawn_turtlebot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', entity_name,
            '-topic', 'robot_description',
            '-x', '-3.0',
            '-y', '-3.0',
            '-z', '0.01',
            '-Y', '0.0'
        ],
        output='screen'
    )
    
    # Maze simulator node
    maze_simulator = Node(
        package='lr_turtlebot_sim',
        executable='maze_simulator',
        name='maze_simulator',
        output='screen',
        parameters=[{
            'maze_name': LaunchConfiguration('world'),
            'start_position_x': -3.0,
            'start_position_y': -3.0
        }]
    )
    
    return LaunchDescription([
        maze_arg,
        robot_arg,
        use_sim_time_arg,
        start_gazebo,
        spawn_turtlebot,
        maze_simulator
    ])