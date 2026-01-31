#!/usr/bin/env python3
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
import torch
import threading
from datetime import datetime

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32, Bool

# Import your PPO agent
try:
    from lr_ppo.ppo_agent import PPOAgent
    PPO_AVAILABLE = True
except ImportError:
    print("ERROR: Cannot import PPOAgent from lr_ppo.ppo_agent")
    print("Make sure your PPO agent is in the correct location")
    PPO_AVAILABLE = False

class PPOTrainer(Node):
    def __init__(self):
        super().__init__('ppo_trainer')
        
        # Training parameters
        self.declare_parameter('max_episodes', 100)
        self.declare_parameter('max_steps_per_episode', 200)
        self.declare_parameter('update_frequency', 256)  # Update every N steps
        self.declare_parameter('save_frequency', 1000)   # Save every N steps
        self.declare_parameter('model_save_path', './models/ppo_turtlebot')
        self.declare_parameter('learning_rate', 0.0003)
        
        # Initialize PPO agent (if available)
        if PPO_AVAILABLE:
            self.agent = PPOAgent(state_dim=29, action_dim=2)
            self.get_logger().info("PPO Agent initialized successfully")
        else:
            self.agent = None
            self.get_logger().error("PPO Agent not available!")
        
        # Subscribers
        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            '/normalized_observation',
            self.obs_callback,
            10
        )
        
        self.reward_sub = self.create_subscription(
            Float32,
            '/current_reward',
            self.reward_callback,
            10
        )
        
        self.done_sub = self.create_subscription(
            Bool,
            '/episode_done',
            self.done_callback,
            10
        )
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_pub = self.create_publisher(Bool, '/request_reset', 10)
        
        # State variables
        self.current_obs = None
        self.current_reward = 0.0
        self.episode_done = False
        
        # Synchronization flags
        self.obs_received = False
        self.reward_received = False
        self.done_received = False
        
        # Training statistics
        self.total_steps = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Create model directory
        model_path = self.get_parameter('model_save_path').value
        os.makedirs(model_path, exist_ok=True)
        
        self.get_logger().info("PPO Trainer initialized")
        
        # Start training after a short delay
        self.training_thread = threading.Thread(target=self.start_training)
        self.training_thread.start()
    
    def obs_callback(self, msg):
        """Callback for observation data"""
        if len(msg.data) == 29:
            self.current_obs = np.array(msg.data, dtype=np.float32)
            self.obs_received = True
        else:
            self.get_logger().warn(f"Observation has wrong size: {len(msg.data)}")
    
    def reward_callback(self, msg):
        """Callback for reward data"""
        self.current_reward = float(msg.data)
        self.reward_received = True
    
    def done_callback(self, msg):
        """Callback for done signal"""
        self.episode_done = bool(msg.data)
        self.done_received = True
    
    def wait_for_observation(self, timeout=3.0):
        """Wait for observation with timeout"""
        start_time = time.time()
        while not self.obs_received and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if self.obs_received:
            obs = self.current_obs.copy()
            self.obs_received = False
            return obs
        else:
            self.get_logger().warn(f"Timeout waiting for observation after {timeout}s")
            return None
    
    def reset_environment(self):
        """Reset the environment and get initial observation"""
        # Reset flags
        self.episode_done = False
        self.done_received = False
        self.reward_received = False
        
        # Send reset signal
        reset_msg = Bool()
        reset_msg.data = True
        self.reset_pub.publish(reset_msg)
        
        # Wait for reset to take effect
        time.sleep(1.0)
        
        # Get initial observation
        self.get_logger().info("Waiting for initial observation after reset...")
        obs = self.wait_for_observation(5.0)
        
        if obs is None:
            self.get_logger().warn("No observation received after reset, using zeros")
            obs = np.zeros(29, dtype=np.float32)
        else:
            self.get_logger().info(f"Observation received: shape={obs.shape}")
        
        return obs
    
    def execute_action(self, action):
        """Execute an action and get next state, reward, done"""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Publish action as cmd_vel
        twist = Twist()
        twist.linear.x = float(action[0] * 0.22)   # Max linear: 0.22 m/s
        twist.angular.z = float(action[1] * 2.84)  # Max angular: 2.84 rad/s
        
        self.get_logger().debug(f"Action: [{action[0]:.3f}, {action[1]:.3f}] -> Cmd: [{twist.linear.x:.3f}, {twist.angular.z:.3f}]")
        
        self.cmd_pub.publish(twist)
        
        # Wait for simulation step (CRITICAL: This must match your simulation rate)
        time.sleep(0.2)
        
        # Get next observation
        next_obs = self.wait_for_observation(1.0)
        if next_obs is None:
            next_obs = self.current_obs if self.current_obs is not None else np.zeros(29)
            reward = -0.1  # Penalty for missing observation
        else:
            reward = self.current_reward if self.reward_received else 0.0
        
        # Check if episode is done
        done = self.episode_done
        
        # Reset flags for next step
        self.reward_received = False
        
        return next_obs, reward, done
    
    def run_episode(self):
        """Run a single training episode"""
        if self.agent is None:
            self.get_logger().error("Cannot run episode: PPO Agent not initialized")
            return 0.0, 0
        
        # Reset environment
        state = self.reset_environment()
        
        episode_reward = 0.0
        episode_steps = 0
        max_steps = self.get_parameter('max_steps_per_episode').value
        
        self.get_logger().info(f"\n=== Starting Episode {self.episode_count + 1} ===")
        
        while not self.episode_done and episode_steps < max_steps:
            # Get action from PPO agent
            action, log_prob, value = self.agent.get_action(state)
            
            # Execute action
            next_state, reward, done = self.execute_action(action)
            
            # Store transition in memory
            self.agent.memory.store(
                state, action, reward, next_state,
                float(done), log_prob, value
            )
            
            # Update for next step
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Update agent periodically
            update_freq = self.get_parameter('update_frequency').value
            if len(self.agent.memory) >= update_freq:
                self.get_logger().info(f"Updating PPO at step {self.total_steps}")
                actor_loss, critic_loss, entropy = self.agent.update()
                self.get_logger().info(f"  Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")
            
            # Save model periodically
            save_freq = self.get_parameter('save_frequency').value
            if self.total_steps % save_freq == 0:
                model_path = os.path.join(
                    self.get_parameter('model_save_path').value,
                    f'ppo_step_{self.total_steps}.pth'
                )
                self.agent.save_model(model_path)
                self.get_logger().info(f"Model saved: {model_path}")
            
            # Log progress every 10 steps
            if episode_steps % 10 == 0:
                self.get_logger().info(f"  Step {episode_steps}: Reward={reward:.3f}, Total={episode_reward:.2f}")
        
        # End of episode
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        
        # Log episode summary
        avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
        
        self.get_logger().info(
            f"\nEpisode {self.episode_count} finished: "
            f"Steps={episode_steps}, "
            f"Total Reward={episode_reward:.2f}, "
            f"Avg Reward (last 10)={avg_reward:.2f}"
        )
        
        # Force update at end of episode if we have data
        if len(self.agent.memory) > 0:
            self.get_logger().info("Performing final update for episode...")
            self.agent.update()
        
        # Save model at end of episode
        model_path = os.path.join(
            self.get_parameter('model_save_path').value,
            f'ppo_episode_{self.episode_count}.pth'
        )
        self.agent.save_model(model_path)
        
        return episode_reward, episode_steps
    
    def start_training(self):
        """Main training loop"""
        if self.agent is None:
            self.get_logger().error("Cannot start training: PPO Agent not initialized")
            return
        
        max_episodes = self.get_parameter('max_episodes').value
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info("STARTING PPO TRAINING")
        self.get_logger().info(f"Max episodes: {max_episodes}")
        self.get_logger().info(f"Max steps per episode: {self.get_parameter('max_steps_per_episode').value}")
        self.get_logger().info(f"Update frequency: {self.get_parameter('update_frequency').value} steps")
        self.get_logger().info(f"Save frequency: {self.get_parameter('save_frequency').value} steps")
        self.get_logger().info(f"Model save path: {self.get_parameter('model_save_path').value}")
        self.get_logger().info(f"{'='*60}\n")
        
        best_reward = -np.inf
        
        try:
            for _ in range(max_episodes):
                episode_reward, episode_steps = self.run_episode()
                
                # Track best performance
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_model_path = os.path.join(
                        self.get_parameter('model_save_path').value,
                        'ppo_best.pth'
                    )
                    self.agent.save_model(best_model_path)
                    self.get_logger().info(f"⭐ New best model saved! Reward: {best_reward:.2f}")
                
                # Early stopping if performance is consistently good
                if len(self.episode_rewards) >= 10:
                    recent_avg = np.mean(self.episode_rewards[-10:])
                    if recent_avg > 20.0:  # Adjust this threshold based on your reward scale
                        self.get_logger().info(f"🎯 Good performance achieved! Average reward: {recent_avg:.2f}")
                        break
                
                # Small delay between episodes
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            self.get_logger().info("\n⚠ Training interrupted by user")
        except Exception as e:
            self.get_logger().error(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Save final model
            final_path = os.path.join(
                self.get_parameter('model_save_path').value,
                'ppo_final.pth'
            )
            self.agent.save_model(final_path)
            
            # Print training summary
            self.print_summary()
            
            self.get_logger().info(f"\n✅ Training completed!")
            self.get_logger().info(f"Final model saved: {final_path}")
    
    def print_summary(self):
        """Print training summary"""
        if not self.episode_rewards:
            self.get_logger().info("No episodes completed")
            return
        
        avg_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        max_reward = np.max(self.episode_rewards)
        min_reward = np.min(self.episode_rewards)
        
        avg_length = np.mean(self.episode_lengths)
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info("TRAINING SUMMARY")
        self.get_logger().info(f"{'='*60}")
        self.get_logger().info(f"Total episodes: {self.episode_count}")
        self.get_logger().info(f"Total steps: {self.total_steps}")
        self.get_logger().info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        self.get_logger().info(f"Best episode: {max_reward:.2f}")
        self.get_logger().info(f"Worst episode: {min_reward:.2f}")
        self.get_logger().info(f"Average episode length: {avg_length:.1f} steps")
        self.get_logger().info(f"{'='*60}")

def main(args=None):
    rclpy.init(args=args)
    
    # Create trainer node
    trainer = PPOTrainer()
    
    try:
        # Keep ROS spinning
        rclpy.spin(trainer)
    except KeyboardInterrupt:
        trainer.get_logger().info("Shutting down trainer...")
    except Exception as e:
        trainer.get_logger().error(f"Trainer error: {e}")
    finally:
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()