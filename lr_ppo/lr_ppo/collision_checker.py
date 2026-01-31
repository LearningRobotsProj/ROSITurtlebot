#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

class CollisionChecker(Node):
    def __init__(self, node_name='collision_checker'):
        super().__init__(node_name)
        
        # Parameters - CRITICAL: Adjust these for TurtleBot3!
        self.declare_parameter('front_collision_threshold', 0.18)  # TurtleBot3 radius ~0.17m
        self.declare_parameter('side_collision_threshold', 0.15)
        self.declare_parameter('goal_threshold', 0.5)
        self.declare_parameter('timeout_steps', 1000)
        self.declare_parameter('collision_angle_range', 30.0)  # degrees
        
        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )
        
        # Publishers
        self.collision_pub = self.create_publisher(Bool, '/collision', 10)
        self.success_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.timeout_pub = self.create_publisher(Bool, '/timeout', 10)
        
        # State
        self.current_position = None
        self.goal_position = None
        self.lidar_ranges = None
        self.lidar_angles = None
        self.step_count = 0
        self.last_collision_time = 0
        
        # Timer for checking
        self.timer = self.create_timer(0.1, self.check_conditions)  # 10Hz
        
        self.get_logger().info("Collision Checker node initialized - TurtleBot3 optimized")
    
    def lidar_callback(self, msg):
        # Store complete scan data
        self.lidar_ranges = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        
        self.get_logger().debug(f"Lidar ranges received: {len(msg.ranges)} samples")
    
    def odom_callback(self, msg):
        self.current_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]
    
    def goal_callback(self, msg):
        self.goal_position = [
            msg.pose.position.x,
            msg.pose.position.y
        ]
        self.get_logger().info(f"Goal position set: {self.goal_position}")
    
    def check_collision(self):
        if self.lidar_ranges is None:
            return False
        
        front_threshold = self.get_parameter('front_collision_threshold').value
        side_threshold = self.get_parameter('side_collision_threshold').value
        collision_angle = np.deg2rad(self.get_parameter('collision_angle_range').value)
        
        # Convert ranges to numpy array for easier processing
        ranges = np.array(self.lidar_ranges)
        
        # Handle invalid values
        valid_mask = ~np.isnan(ranges) & ~np.isinf(ranges) & (ranges > 0)
        if not np.any(valid_mask):
            return False
        
        valid_ranges = ranges[valid_mask]
        
        # Get angles for each range
        angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)
        valid_angles = angles[valid_mask]
        
        # Check front collision (most important for TurtleBot3)
        front_mask = np.abs(valid_angles) < collision_angle / 2
        if np.any(front_mask):
            front_ranges = valid_ranges[front_mask]
            min_front_distance = np.min(front_ranges)
            
            if min_front_distance < front_threshold:
                self.get_logger().warn(
                    f"Front collision! Min distance: {min_front_distance:.3f}m "
                    f"(threshold: {front_threshold:.3f}m)"
                )
                return True
        
        # Check side collisions
        side_mask = (np.abs(valid_angles) > np.pi/4) & (np.abs(valid_angles) < 3*np.pi/4)
        if np.any(side_mask):
            side_ranges = valid_ranges[side_mask]
            min_side_distance = np.min(side_ranges)
            
            if min_side_distance < side_threshold:
                self.get_logger().warn(
                    f"Side collision! Min distance: {min_side_distance:.3f}m "
                    f"(threshold: {side_threshold:.3f}m)"
                )
                return True
        
        # Overall minimum distance check (safety)
        min_distance = np.min(valid_ranges)
        if min_distance < 0.12:  # Absolute minimum safety distance
            self.get_logger().warn(
                f"Safety collision! Min distance: {min_distance:.3f}m"
            )
            return True
        
        return False
    
    def check_goal_reached(self):
        if self.current_position is None or self.goal_position is None:
            return False
        
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        goal_threshold = self.get_parameter('goal_threshold').value
        reached = distance < goal_threshold
        
        if reached:
            self.get_logger().info(f"Goal reached! Distance: {distance:.3f}m")
        
        return reached
    
    def check_timeout(self):
        timeout_steps = self.get_parameter('timeout_steps').value
        timeout = self.step_count >= timeout_steps
        
        if timeout:
            self.get_logger().info(f"Timeout after {self.step_count} steps")
        
        return timeout
    
    def check_conditions(self):
        self.step_count += 1
        
        # Check all conditions
        collision = self.check_collision()
        goal_reached = self.check_goal_reached()
        timeout = self.check_timeout()
        
        # Publish collision status
        collision_msg = Bool()
        collision_msg.data = collision
        self.collision_pub.publish(collision_msg)
        
        # Publish goal reached status
        success_msg = Bool()
        success_msg.data = goal_reached
        self.success_pub.publish(success_msg)
        
        # Publish timeout status
        timeout_msg = Bool()
        timeout_msg.data = timeout
        self.timeout_pub.publish(timeout_msg)
        
        # Log status periodically
        if self.step_count % 50 == 0:
            if self.lidar_ranges is not None:
                valid_ranges = [r for r in self.lidar_ranges if not (np.isinf(r) or np.isnan(r) or r == 0)]
                if valid_ranges:
                    min_dist = min(valid_ranges)
                    self.get_logger().info(
                        f"Step {self.step_count}: Min lidar distance = {min_dist:.3f}m"
                    )
        
        # Reset if episode ended
        if collision or goal_reached or timeout:
            self.get_logger().info(
                f"Episode ended: collision={collision}, "
                f"goal_reached={goal_reached}, timeout={timeout}"
            )
            self.step_count = 0
            self.last_collision_time = self.step_count

def main(args=None):
    rclpy.init(args=args)
    node = CollisionChecker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Collision checker stopped by user")
    except Exception as e:
        node.get_logger().error(f"Collision checker error: {str(e)}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()