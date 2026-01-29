#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MovementTester(Node):
    def __init__(self):
        super().__init__('movement_tester')
        
        self.get_logger().info("=== MOVEMENT TESTER STARTING ===")
        
        # Create publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Wait a moment for publisher to be ready
        time.sleep(2.0)
        
        # Test robot movement
        self.test_movement()
        
    def test_movement(self):
        """Test if robot can move in Gazebo"""
        self.get_logger().info("Testing robot movement in Gazebo...")
        
        # Test sequence
        tests = [
            ("Moving FORWARD", 0.2, 0.0),
            ("Turning RIGHT", 0.0, -0.5),
            ("Turning LEFT", 0.0, 0.5),
            ("Moving in CIRCLE", 0.1, 0.3),
        ]
        
        for test_name, linear, angular in tests:
            self.get_logger().info(f"Test: {test_name}")
            
            # Create twist message
            twist = Twist()
            twist.linear.x = float(linear)
            twist.angular.z = float(angular)
            
            # Publish for 3 seconds
            start_time = time.time()
            while time.time() - start_time < 3.0:
                self.cmd_pub.publish(twist)
                time.sleep(0.1)
            
            # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            time.sleep(1.0)
        
        self.get_logger().info("=== MOVEMENT TEST COMPLETE ===")
        self.get_logger().info("If robot moved in Gazebo, training will start soon...")
        
        # Keep node alive for a bit
        time.sleep(5.0)

def main(args=None):
    rclpy.init(args=args)
    node = MovementTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()