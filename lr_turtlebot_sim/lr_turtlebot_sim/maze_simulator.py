#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import time

class MazeSimulator(Node):
    def __init__(self):
        super().__init__('maze_simulator')

        # Parameters
        self.declare_parameter('robot_name', 'turtlebot3_burger')
        self.declare_parameter('start_x', -3.0)
        self.declare_parameter('start_y', -3.0)
        self.declare_parameter('goal_x', 3.5)
        self.declare_parameter('goal_y', 3.5)

        # Publishers / Subscribers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.reset_done_pub = self.create_publisher(Bool, '/reset_episode', 10)
        self.create_subscription(Bool, '/request_reset', self.reset_cb, 10)

        # Service client (Gazebo Sim)
        self.set_state_client = self.create_client(
            SetEntityPose,
            '/world/maze1/set_pose'
        )

        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gazebo set_pose service...')

        self.get_logger().info('MazeSimulator started')

        time.sleep(3.0)
        self.reset_robot()
        self.publish_goal()

    def reset_robot(self):
        self.get_logger().info('Resetting robot pose...')

        req = SetEntityPose.Request()
        req.name = self.get_parameter('robot_name').value

        req.pose.position.x = self.get_parameter('start_x').value
        req.pose.position.y = self.get_parameter('start_y').value
        req.pose.position.z = 0.05

        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = 0.0
        req.pose.orientation.w = 1.0

        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            self.get_logger().info('Robot pose reset completed')
        else:
            self.get_logger().warn('Could not reset robot pose')

    def publish_goal(self):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = self.get_parameter('goal_x').value
        goal.pose.position.y = self.get_parameter('goal_y').value
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        self.get_logger().info(
            f'Published goal at ({goal.pose.position.x}, {goal.pose.position.y})'
        )

    def reset_cb(self, msg):
        if msg.data:
            self.get_logger().info('Received reset request')
            self.reset_robot()
            self.publish_goal()

            done = Bool()
            done.data = True
            self.reset_done_pub.publish(done)
            self.get_logger().info('Episode reset done')

def main():
    rclpy.init()
    node = MazeSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
