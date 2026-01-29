import gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool

from observation_vector import build_observation
from reward_function import compute_reward
from collision_checker import check_collision

class TurtlebotGymEnv(gym.Env, Node):

    def __init__(self):
        Node.__init__(self, 'turtlebot_gym_env')
        gym.Env.__init__(self)

        self.get_logger().info("Initializing Turtlebot Gym Environment")

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([0.22, 1.0]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(362,),
            dtype=np.float32
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_pub = self.create_publisher(Bool, '/request_reset', 10)

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)

        self.laser = None
        self.odom = None
        self.goal = None

        self.prev_dist = None

    def scan_cb(self, msg):
        self.laser = np.array(msg.ranges)

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = 0.0
        self.odom = [x, y, theta]

    def goal_cb(self, msg):
        self.goal = [msg.pose.position.x, msg.pose.position.y]

    def reset(self):
        self.get_logger().info("Resetting environment")

        msg = Bool()
        msg.data = True
        self.reset_pub.publish(msg)

        while self.laser is None or self.odom is None or self.goal is None:
            rclpy.spin_once(self)

        self.prev_dist = np.hypot(
            self.goal[0] - self.odom[0],
            self.goal[1] - self.odom[1]
        )

        obs = build_observation(self.laser, self.odom, self.goal)
        return obs

    def step(self, action):
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)

        rclpy.spin_once(self, timeout_sec=0.05)

        collision = check_collision(self.laser)
        curr_dist = np.hypot(
            self.goal[0] - self.odom[0],
            self.goal[1] - self.odom[1]
        )

        reached = curr_dist < 0.3
        reward = compute_reward(self.prev_dist, curr_dist, collision, reached)

        self.prev_dist = curr_dist
        done = collision or reached

        obs = build_observation(self.laser, self.odom, self.goal)

        self.get_logger().debug(
            f"Step | dist={curr_dist:.2f}, reward={reward:.2f}, done={done}"
        )

        return obs, reward, done, {}
