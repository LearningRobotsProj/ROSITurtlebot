# lr_ppo/lr_ppo/collision_checker.py
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
        
        # Parameters
        self.declare_parameter('collision_threshold', 0.15)
        self.declare_parameter('goal_threshold', 0.3)
        self.declare_parameter('timeout_steps', 1000)
        
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
        self.step_count = 0
        
        # Timer for checking
        self.timer = self.create_timer(0.1, self.check_conditions)
        
        self.get_logger().info("Collision Checker node initialized")
    
    def lidar_callback(self, msg):
        # Get valid ranges
        valid_ranges = []
        for r in msg.ranges:
            if not (np.isinf(r) or np.isnan(r)):
                valid_ranges.append(r)
        
        if valid_ranges:
            self.lidar_ranges = valid_ranges
    
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
    
    def check_collision(self):
        if self.lidar_ranges is None:
            return False
        
        collision_threshold = self.get_parameter('collision_threshold').value
        min_distance = min(self.lidar_ranges)
        
        collision = min_distance < collision_threshold
        
        if collision:
            self.get_logger().warn(f"Collision! Min distance: {min_distance:.3f}m")
        
        return collision
    
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
        
        collision = self.check_collision()
        goal_reached = self.check_goal_reached()
        timeout = self.check_timeout()
        
        # Publish results
        collision_msg = Bool()
        collision_msg.data = collision
        self.collision_pub.publish(collision_msg)
        
        success_msg = Bool()
        success_msg.data = goal_reached
        self.success_pub.publish(success_msg)
        
        timeout_msg = Bool()
        timeout_msg.data = timeout
        self.timeout_pub.publish(timeout_msg)
        
        # Reset if episode ended
        if collision or goal_reached or timeout:
            self.step_count = 0
    
    def reset(self):
        self.step_count = 0

def main(args=None):
    rclpy.init(args=args)
    node = CollisionChecker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def debug_main():
    """Debug function to test collision checker."""
    print("=== Testing Collision Checker ===")
    
    node = CollisionChecker('test_collision')
    
    # Test 1: Collision detection
    print("\n1. Testing collision detection...")
    node.lidar_ranges = [0.1, 0.5, 1.0]
    collision = node.check_collision()
    
    if collision:
        print("✓ Collision correctly detected")
    else:
        print("✗ Collision NOT detected")
    
    # Test 2: No collision
    print("\n2. Testing no collision...")
    node.lidar_ranges = [0.5, 0.6, 0.7]
    collision = node.check_collision()
    
    if not collision:
        print("✓ No collision correctly detected")
    else:
        print("✗ False collision detected")
    
    # Test 3: Goal reached
    print("\n3. Testing goal reached...")
    node.current_position = [0.0, 0.0]
    node.goal_position = [0.2, 0.2]
    goal_reached = node.check_goal_reached()
    
    if goal_reached:
        print("✓ Goal reached correctly detected")
    else:
        print("✗ Goal reached NOT detected")
    
    # Test 4: Goal not reached
    print("\n4. Testing goal not reached...")
    node.current_position = [0.0, 0.0]
    node.goal_position = [2.0, 2.0]
    goal_reached = node.check_goal_reached()
    
    if not goal_reached:
        print("✓ Goal not reached correctly detected")
    else:
        print("✗ False goal reached detected")
    
    return True

if __name__ == "__main__":
    main()