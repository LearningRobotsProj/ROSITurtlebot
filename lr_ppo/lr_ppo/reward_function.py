# lr_ppo/lr_ppo/reward_function.py
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
        self.declare_parameter('reward_distance_multiplier', 0.1)
        self.declare_parameter('penalty_time_step', -0.05)
        self.declare_parameter('penalty_excessive_turning', -0.01)
        self.declare_parameter('max_angular_vel_for_turn_penalty', 0.5)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )
        self.collision_sub = self.create_subscription(
            Bool, '/collision', self.collision_callback, 10
        )
        self.success_sub = self.create_subscription(
            Bool, '/goal_reached', self.success_callback, 10
        )
        
        # Publishers
        self.reward_pub = self.create_publisher(Float32, '/current_reward', 10)
        self.cumulative_reward_pub = self.create_publisher(Float32, '/cumulative_reward', 10)
        
        # State
        self.current_position = None
        self.goal_position = None
        self.collision_detected = False
        self.goal_reached = False
        self.cumulative_reward = 0.0
        self.previous_distance = None
        self.episode_step = 0
        
        self.get_logger().info("Reward Function node initialized")
    
    def odom_callback(self, msg):
        self.current_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]
        self.current_angular_vel = abs(msg.twist.twist.angular.z)
    
    def goal_callback(self, msg):
        self.goal_position = [
            msg.pose.position.x,
            msg.pose.position.y
        ]
        # Reset previous distance when goal changes
        self.previous_distance = None
    
    def collision_callback(self, msg):
        self.collision_detected = msg.data
    
    def success_callback(self, msg):
        self.goal_reached = msg.data
    
    def compute_distance_to_goal(self):
        if self.current_position is None or self.goal_position is None:
            return None
        
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        return np.sqrt(dx**2 + dy**2)
    
    def compute_progress_reward(self):
        current_distance = self.compute_distance_to_goal()
        
        if current_distance is None or self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        distance_change = self.previous_distance - current_distance
        progress_reward = distance_change * self.get_parameter('reward_distance_multiplier').value
        
        self.previous_distance = current_distance
        return progress_reward
    
    def compute_turn_penalty(self):
        max_angular = self.get_parameter('max_angular_vel_for_turn_penalty').value
        if hasattr(self, 'current_angular_vel') and self.current_angular_vel > max_angular:
            return self.get_parameter('penalty_excessive_turning').value
        return 0.0
    
    def compute_time_penalty(self):
        return self.get_parameter('penalty_time_step').value
    
    def compute_reward(self):
        self.episode_step += 1
        
        reward = 0.0
        done = False
        
        # Check termination conditions
        if self.collision_detected:
            reward = self.get_parameter('penalty_collision').value
            done = True
            self.get_logger().warn(f"Collision! Reward: {reward}")
        
        elif self.goal_reached:
            reward = self.get_parameter('reward_goal_reached').value
            done = True
            self.get_logger().info(f"Goal reached! Reward: {reward}")
        
        else:
            # Normal step rewards
            reward += self.compute_progress_reward()
            reward += self.compute_turn_penalty()
            reward += self.compute_time_penalty()
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Publish rewards
        self.publish_rewards(reward)
        
        return reward, done
    
    def publish_rewards(self, step_reward):
        reward_msg = Float32()
        reward_msg.data = float(step_reward)
        self.reward_pub.publish(reward_msg)
        
        cum_reward_msg = Float32()
        cum_reward_msg.data = float(self.cumulative_reward)
        self.cumulative_reward_pub.publish(cum_reward_msg)
    
    def reset(self):
        self.episode_step = 0
        self.previous_distance = None
        self.collision_detected = False
        self.goal_reached = False
        self.cumulative_reward = 0.0

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

def debug_main():
    """Debug function to test reward function."""
    print("=== Testing Reward Function ===")
    
    node = RewardFunction('test_reward')
    
    # Test 1: Goal reached
    print("\n1. Testing goal reached...")
    node.goal_reached = True
    reward, done = node.compute_reward()
    
    if reward == 100.0 and done:
        print("✓ Goal reward correct")
    else:
        print(f"✗ Goal reward incorrect: {reward}, done={done}")
    
    # Reset
    node.reset()
    
    # Test 2: Collision
    print("\n2. Testing collision...")
    node.collision_detected = True
    reward, done = node.compute_reward()
    
    if reward == -50.0 and done:
        print("✓ Collision penalty correct")
    else:
        print(f"✗ Collision penalty incorrect: {reward}, done={done}")
    
    # Reset
    node.reset()
    
    # Test 3: Progress reward
    print("\n3. Testing progress reward...")
    node.current_position = [0.0, 0.0]
    node.goal_position = [10.0, 0.0]
    node.previous_distance = 10.0  # Start 10m away
    node.current_position = [1.0, 0.0]  # Move 1m closer
    
    reward, done = node.compute_reward()
    expected = 0.9  # 1.0 * 0.1 - 0.05 time penalty - 0.01 turn penalty approx
    
    if abs(reward - expected) < 0.1:
        print(f"✓ Progress reward approximately correct: {reward:.2f}")
    else:
        print(f"✗ Progress reward incorrect: {reward:.2f} (expected ~{expected:.2f})")
    
    return True

if __name__ == "__main__":
    main()