# lr_turtlebot_sim/lr_turtlebot_sim/maze_simulator.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Bool
import numpy as np
import time

class MazeSimulator(Node):
    def __init__(self):
        super().__init__('maze_simulator')
        
        # Parameters
        self.declare_parameter('maze_name', 'maze1')
        self.declare_parameter('goal_position_x', 3.0)
        self.declare_parameter('goal_position_y', 3.0)
        self.declare_parameter('start_position_x', -3.0)
        self.declare_parameter('start_position_y', -3.0)
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.episode_reset_pub = self.create_publisher(Bool, '/reset_episode', 10)
        
        # Subscribers
        self.create_subscription(Bool, '/request_reset', self.reset_callback, 10)
        
        # Set initial positions
        self.set_initial_pose()
        time.sleep(1.0)
        self.set_goal()
        
        self.get_logger().info("Maze Simulator initialized")
    
    def set_initial_pose(self):
        """Set the initial robot pose."""
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.header.frame_id = 'map'
        
        # Get start position from parameters
        start_x = self.get_parameter('start_position_x').value
        start_y = self.get_parameter('start_position_y').value
        
        initial_pose.pose.pose.position.x = start_x
        initial_pose.pose.pose.position.y = start_y
        initial_pose.pose.pose.orientation.w = 1.0
        
        # Set some covariance
        initial_pose.pose.covariance[0] = 0.25
        initial_pose.pose.covariance[7] = 0.25
        initial_pose.pose.covariance[35] = 0.06853891945200942
        
        self.initial_pose_pub.publish(initial_pose)
        self.get_logger().info(f"Initial pose set to ({start_x}, {start_y})")
    
    def set_goal(self):
        """Set the goal position based on maze."""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        
        maze = self.get_parameter('maze_name').value
        
        # Set goal based on maze
        if maze == 'maze1':
            goal_msg.pose.position.x = 3.5
            goal_msg.pose.position.y = 3.5
        elif maze == 'maze2':
            goal_msg.pose.position.x = -3.5
            goal_msg.pose.position.y = -3.5
        elif maze == 'maze3':
            goal_msg.pose.position.x = 4.5
            goal_msg.pose.position.y = -4.5
        else:
            # Use parameter values
            goal_msg.pose.position.x = self.get_parameter('goal_position_x').value
            goal_msg.pose.position.y = self.get_parameter('goal_position_y').value
        
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"Goal set at ({goal_msg.pose.position.x}, {goal_msg.pose.position.y})")
        
        return goal_msg
    
    def reset_callback(self, msg):
        """Handle reset request."""
        if msg.data:
            self.get_logger().info("Resetting episode...")
            
            # Reset robot pose
            self.set_initial_pose()
            time.sleep(0.5)
            
            # Set new goal (could be same or different)
            self.set_goal()
            
            # Publish reset completion
            reset_done = Bool()
            reset_done.data = True
            self.episode_reset_pub.publish(reset_done)

def main(args=None):
    rclpy.init(args=args)
    simulator = MazeSimulator()
    rclpy.spin(simulator)
    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()