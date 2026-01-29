#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.get_logger().info("Test node started")
        self.timer = self.create_timer(1.0, self.timer_callback)
    
    def timer_callback(self):
        self.get_logger().info("Test node running...")

def main():
    rclpy.init()
    node = TestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()