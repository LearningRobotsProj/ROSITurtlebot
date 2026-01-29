#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class RewardFunction(Node):
    def __init__(self, node_name='reward_function'):
        super().__init__(node_name)
        
        # Reward parameters
        self.declare_parameter('reward_goal_reached', 100.0)
        self.declare_parameter('penalty_collision', -50.0)
        self.declare_parameter('reward_distance_multiplier', 5.0)  # Increased
        self.declare_parameter('penalty_time_step', -0.1)
        self.declare_parameter('penalty_excessive_turning', -0.05)
        self.declare_parameter('max_angular_vel_for_turn_penalty', 1.0)
        self.declare_parameter('goal_threshold', 0.3)  # meters
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.collision_sub = self.create_subscription(Bool, '/collision', self.collision_callback, 10)
        self.reset_sub = self.create_subscription(Bool, '/request_reset', self.reset_callback, 10)
        
        # Publishers
        self.reward_pub = self.create_publisher(Float32, '/current_reward', 10)
        self.cumulative_reward_pub = self.create_publisher(Float32, '/cumulative_reward', 10)
        self.episode_done_pub = self.create_publisher(Bool, '/episode_done', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        
        # State
        self.current_position = None
        self.current_orientation = 0.0
        self.current_angular_vel = 0.0
        self.goal_position = None
        self.collision_detected = False
        self.episode_active = False
        self.cumulative_reward = 0.0
        self.previous_distance = None
        self.episode_step = 0
        
        # Timer for continuous reward computation
        self.reward_timer = self.create_timer(0.1, self.compute_and_publish_reward)  # 10 Hz
        
        self.get_logger().info("Reward Function node initialized - Waiting for reset signal...")
    
    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_callback(self, msg):
        self.current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = np.arctan2(siny_cosp, cosy_cosp)
        
        self.current_angular_vel = abs(msg.twist.twist.angular.z)
    
    def goal_callback(self, msg):
        self.goal_position = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().info(f"Goal set at: {self.goal_position}")
        
        if self.episode_active and self.previous_distance is None:
            self.previous_distance = self.compute_distance_to_goal()
    
    def collision_callback(self, msg):
        if msg.data and self.episode_active:
            self.collision_detected = True
            self.get_logger().warn("Collision detected!")
    
    def reset_callback(self, msg):
        if msg.data:
            self.reset_episode()
    
    # -----------------------------
    # Reward computation
    # -----------------------------
    def compute_distance_to_goal(self):
        if self.current_position is None or self.goal_position is None:
            return None
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        return np.sqrt(dx**2 + dy**2)
    
    def check_goal_reached(self):
        if self.current_position is None or self.goal_position is None:
            return False
        
        distance = self.compute_distance_to_goal()
        threshold = self.get_parameter('goal_threshold').value
        return distance < threshold
    
    def compute_progress_reward(self):
        if not self.episode_active:
            return 0.0
            
        current_distance = self.compute_distance_to_goal()
        if current_distance is None or self.previous_distance is None:
            return 0.0
        
        # Reward for reducing distance to goal
        progress = self.previous_distance - current_distance
        reward = progress * self.get_parameter('reward_distance_multiplier').value
        
        # Bonus for making progress
        if progress > 0:
            reward += 0.1
        
        self.previous_distance = current_distance
        return reward
    
    def compute_turn_penalty(self):
        if not self.episode_active:
            return 0.0
            
        max_vel = self.get_parameter('max_angular_vel_for_turn_penalty').value
        if self.current_angular_vel > max_vel:
            penalty = self.get_parameter('penalty_excessive_turning').value
            # Scale penalty with angular velocity
            return penalty * (self.current_angular_vel / max_vel)
        return 0.0
    
    def compute_reward(self):
        if not self.episode_active:
            return 0.0, False
        
        reward = 0.0
        done = False
        
        # Check termination conditions
        goal_reached = self.check_goal_reached()
        
        if self.collision_detected:
            reward = self.get_parameter('penalty_collision').value
            done = True
            self.get_logger().warn(f"Collision penalty: {reward}")
            
        elif goal_reached:
            reward = self.get_parameter('reward_goal_reached').value
            done = True
            
            # Publish goal reached
            goal_msg = Bool()
            goal_msg.data = True
            self.goal_reached_pub.publish(goal_msg)
            
            self.get_logger().info(f"Goal reached! Reward: {reward}")
            
        else:
            # Normal step rewards/penalties
            reward += self.compute_progress_reward()
            reward += self.compute_turn_penalty()
            reward += self.get_parameter('penalty_time_step').value
            
            # Penalty for being stuck
            if self.episode_step > 100 and abs(reward) < 0.01:
                reward -= 0.1
        
        self.episode_step += 1
        
        # Timeout after too many steps
        if self.episode_step > 500:
            if not done:
                done = True
                reward -= 10.0  # Timeout penalty
                self.get_logger().info("Episode timeout")
        
        return reward, done
    
    def compute_and_publish_reward(self):
        if not self.episode_active:
            return
        
        reward, done = self.compute_reward()
        
        # Publish current reward
        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.reward_pub.publish(reward_msg)
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Publish cumulative reward
        cum_msg = Float32()
        cum_msg.data = float(self.cumulative_reward)
        self.cumulative_reward_pub.publish(cum_msg)
        
        # Publish episode done if needed
        if done:
            done_msg = Bool()
            done_msg.data = True
            self.episode_done_pub.publish(done_msg)
            
            # Log episode summary
            self.get_logger().info(
                f"Episode ended: Step {self.episode_step}, "
                f"Final reward: {reward:.2f}, "
                f"Total: {self.cumulative_reward:.2f}"
            )
            
            # Reset episode
            self.episode_active = False
    
    # -----------------------------
    # Episode management
    # -----------------------------
    def reset_episode(self):
        self.episode_step = 0
        self.collision_detected = False
        self.cumulative_reward = 0.0
        self.previous_distance = self.compute_distance_to_goal()
        self.episode_active = True
        
        self.get_logger().info("Episode reset - Starting new episode")
        
        # Publish initial reward (0)
        reward_msg = Float32()
        reward_msg.data = 0.0
        self.reward_pub.publish(reward_msg)

# -----------------------------
# Entry point
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RewardFunction()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()