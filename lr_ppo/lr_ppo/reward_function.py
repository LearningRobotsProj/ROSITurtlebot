#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import math
import time

class RewardFunction(Node):
    def __init__(self, node_name='reward_function'):
        super().__init__(node_name)
        
        # **PROPERLY SCALED REWARD PARAMETERS**
        self.declare_parameter('reward_goal_reached', 50.0)      # Reduced from 100
        self.declare_parameter('penalty_collision', -20.0)       # Reduced from -50
        self.declare_parameter('reward_distance_multiplier', 1.0)  # SIGNIFICANTLY REDUCED
        self.declare_parameter('penalty_time_step', -0.02)       # Slightly increased
        self.declare_parameter('max_steps', 100)                 # Shorter episodes
        self.declare_parameter('goal_threshold', 0.5)
        self.declare_parameter('angle_reward_multiplier', 0.1)   # SIGNIFICANTLY REDUCED
        self.declare_parameter('velocity_bonus', 0.005)          # SIGNIFICANTLY REDUCED
        self.declare_parameter('stuck_penalty_threshold', 15)    # Steps without progress
        
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
        self.current_linear_vel = 0.0
        self.goal_position = None
        self.collision_detected = False
        self.episode_active = False
        self.cumulative_reward = 0.0
        self.previous_distance = None
        self.initial_distance = None
        self.episode_step = 0
        self.steps_without_progress = 0
        self.last_progress_distance = None
        self.episode_start_time = None
        
        # Timer for reward computation
        self.reward_timer = self.create_timer(0.1, self.compute_and_publish_reward)
        
        self.get_logger().info("✅ FIXED Reward Function - Properly scaled rewards")
        self.get_logger().warning("⚠ IMPORTANT: Rewards are now SMALL (-0.1 to 0.2 range)")
    
    def odom_callback(self, msg):
        self.current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)
        
        self.current_linear_vel = abs(msg.twist.twist.linear.x)
    
    def goal_callback(self, msg):
        self.goal_position = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().info(f"🎯 Goal set at: {self.goal_position}")
        
        if self.episode_active and self.previous_distance is None:
            self.previous_distance = self.compute_distance_to_goal()
            self.initial_distance = self.previous_distance
            self.last_progress_distance = self.previous_distance
            self.get_logger().info(f"📏 Initial distance to goal: {self.initial_distance:.2f}m")
    
    def collision_callback(self, msg):
        if msg.data and self.episode_active and not self.collision_detected:
            self.collision_detected = True
            self.get_logger().warn("💥 COLLISION DETECTED! Ending episode.")
    
    def reset_callback(self, msg):
        if msg.data:
            self.get_logger().info("🔄 Received reset signal")
            self.reset_episode()
    
    def compute_distance_to_goal(self):
        if self.current_position is None or self.goal_position is None:
            return None
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        return math.sqrt(dx**2 + dy**2)
    
    def compute_angle_to_goal(self):
        if self.current_position is None or self.goal_position is None:
            return 0.0
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        goal_angle = math.atan2(dy, dx)
        angle_diff = goal_angle - self.current_orientation
        # Normalize to [-pi, pi]
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        return angle_diff
    
    def check_goal_reached(self):
        if self.current_position is None or self.goal_position is None:
            return False
        distance = self.compute_distance_to_goal()
        threshold = self.get_parameter('goal_threshold').value
        reached = distance < threshold
        
        if reached:
            self.get_logger().info(f"🎉 GOAL REACHED! Distance: {distance:.2f}m")
        
        return reached
    
    def compute_reward(self):
        """Compute reward for current step - FIXED TO GIVE MEANINGFUL REWARDS"""
        if not self.episode_active:
            return 0.0, False
        
        reward = 0.0
        done = False
        
        # Check termination conditions FIRST
        if self.collision_detected:
            reward = self.get_parameter('penalty_collision').value
            done = True
            self.get_logger().warn(f"💥 Episode ended: Collision! Reward: {reward}")
        
        elif self.check_goal_reached():
            reward = self.get_parameter('reward_goal_reached').value
            done = True
            
            goal_msg = Bool()
            goal_msg.data = True
            self.goal_reached_pub.publish(goal_msg)
            
            self.get_logger().info(f"🎉 Episode ended: Goal reached! Reward: {reward}")
        
        else:
            # **NORMAL STEP REWARDS - CRITICAL FIXES**
            current_distance = self.compute_distance_to_goal()
            
            if current_distance is not None and self.previous_distance is not None:
                # 1. PROGRESS REWARD (most important)
                distance_change = self.previous_distance - current_distance
                
                if distance_change > 0.02:  # At least 2cm progress
                    progress_reward = distance_change * self.get_parameter('reward_distance_multiplier').value
                    reward += progress_reward
                    self.steps_without_progress = 0
                    self.last_progress_distance = current_distance
                    
                    # Small bonus for significant progress
                    if distance_change > 0.1:
                        reward += 0.05
                        
                elif distance_change < -0.02:  # Moving away
                    penalty = distance_change * self.get_parameter('reward_distance_multiplier').value * 2
                    reward += penalty
                    self.steps_without_progress += 1
                    
                else:  # Little movement (±2cm)
                    reward += 0.0  # No progress reward
                    self.steps_without_progress += 1
                
                self.previous_distance = current_distance
            
            # 2. ANGLE REWARD (small bonus for facing goal)
            angle_to_goal = abs(self.compute_angle_to_goal())
            if angle_to_goal < math.pi/6:  # Within 30 degrees
                angle_reward = (1.0 - angle_to_goal/(math.pi/6)) * self.get_parameter('angle_reward_multiplier').value
                reward += angle_reward
            
            # 3. STUCK PENALTY (if no progress for a while)
            stuck_threshold = self.get_parameter('stuck_penalty_threshold').value
            if self.steps_without_progress > stuck_threshold:
                penalty = -0.02 * (self.steps_without_progress - stuck_threshold)
                reward += penalty
            
            # 4. TIME PENALTY (encourage efficiency)
            reward += self.get_parameter('penalty_time_step').value
            
            # 5. VELOCITY BONUS (small bonus for moving forward)
            if self.current_linear_vel > 0.05:
                velocity_bonus = self.current_linear_vel * self.get_parameter('velocity_bonus').value
                reward += velocity_bonus
        
        self.episode_step += 1
        
        # Timeout after max steps (FORCE episode end)
        max_steps = self.get_parameter('max_steps').value
        if self.episode_step >= max_steps and not done:
            done = True
            # Penalty based on progress made
            if self.initial_distance and current_distance:
                progress = 1.0 - (current_distance / self.initial_distance)
                if progress > 0.3:  # Made decent progress
                    reward = 5.0 * progress  # Small positive based on progress
                else:
                    reward = -5.0 * (1.0 - progress)  # Penalty based on lack of progress
            else:
                reward = -5.0  # Default timeout penalty
            self.get_logger().info(f"⏰ Episode ended: Timeout at {self.episode_step} steps. Reward: {reward:.2f}")
        
        return reward, done
    
    def compute_and_publish_reward(self):
        if not self.episode_active:
            return
        
        reward, done = self.compute_reward()
        
        # Publish current reward
        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.reward_pub.publish(reward_msg)
        
        # Update cumulative
        self.cumulative_reward += reward
        
        # Publish cumulative
        cum_msg = Float32()
        cum_msg.data = float(self.cumulative_reward)
        self.cumulative_reward_pub.publish(cum_msg)
        
        # Log progress every 5 steps (more frequent for debugging)
        if self.episode_step % 5 == 0:
            current_distance = self.compute_distance_to_goal()
            if current_distance and self.initial_distance:
                progress = 1.0 - (current_distance / self.initial_distance)
                self.get_logger().info(
                    f"📊 Step {self.episode_step}: "
                    f"Dist={current_distance:.2f}m ({progress:.1%}), "
                    f"Reward={reward:.3f}, "
                    f"Total={self.cumulative_reward:.2f}"
                )
        
        # Handle episode end
        if done:
            done_msg = Bool()
            done_msg.data = True
            self.episode_done_pub.publish(done_msg)
            
            # Log episode summary
            final_distance = self.compute_distance_to_goal()
            if final_distance and self.initial_distance:
                progress = 1.0 - (final_distance / self.initial_distance)
                self.get_logger().info(
                    f"🏁 Episode END: Steps={self.episode_step}, "
                    f"Final distance: {final_distance:.2f}m ({progress:.1%}), "
                    f"Total reward: {self.cumulative_reward:.2f}"
                )
            
            # Reset for next episode
            self.episode_active = False
    
    def reset_episode(self):
        self.episode_step = 0
        self.collision_detected = False
        self.cumulative_reward = 0.0
        self.steps_without_progress = 0
        self.previous_distance = self.compute_distance_to_goal()
        self.initial_distance = self.previous_distance
        self.last_progress_distance = self.previous_distance
        self.episode_active = True
        self.episode_start_time = time.time()
        
        if self.initial_distance is not None:
            self.get_logger().info(f"🔄 Episode RESET: Initial distance: {self.initial_distance:.2f}m")
        else:
            self.get_logger().info("🔄 Episode RESET: Waiting for position/goal data")

def main(args=None):
    rclpy.init(args=args)
    node = RewardFunction()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("⏹️ Reward function stopped by user")
    except Exception as e:
        node.get_logger().error(f"❌ Reward function error: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()