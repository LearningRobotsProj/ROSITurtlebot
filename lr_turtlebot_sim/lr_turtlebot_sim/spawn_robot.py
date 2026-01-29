#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import subprocess
import os
import ament_index_python.packages as ament


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('spawn_turtlebot')

    
    tb3_gazebo_pkg = ament.get_package_share_directory('turtlebot3_gazebo')
    model_path = f"{tb3_gazebo_pkg}/models/turtlebot3_burger/model.sdf"

    if not os.path.exists(model_path):
        node.get_logger().error('TurtleBot3 model not found')
        return

    node.get_logger().info('Spawning TurtleBot3 using ros_gz_sim create')

    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-name', 'turtlebot3_burger',
        '-file', model_path,
        '-x', '-3.0',
        '-y', '-3.0',
        '-z', '0.05'
    ]

    subprocess.run(cmd, check=True)

    node.get_logger().info('TurtleBot3 spawned successfully')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
