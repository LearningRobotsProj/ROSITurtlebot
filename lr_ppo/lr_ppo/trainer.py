#!/usr/bin/env python3
import os
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32, Bool


# ===============================
# ROS2 + GYM ENVIRONMENT
# ===============================
class ROS2RobotEnv(gym.Env, Node):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=500):
        Node.__init__(self, "ros2_gym_env")
        gym.Env.__init__(self)

        # ---- Spaces ----
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ---- State ----
        self.current_obs = np.zeros(29, dtype=np.float32)
        self.current_reward = 0.0
        self.done_received = False
        self.current_step = 0
        self.max_steps = max_steps
        
        # Synchronization
        self.obs_event = threading.Event()
        self.reward_event = threading.Event()
        self.done_event = threading.Event()
        
        # Store latest data
        self.latest_obs = None
        self.latest_reward = 0.0
        self.latest_done = False

        # ---- ROS Interfaces ----
        print("\nCreating subscribers...")
        
        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            '/normalized_observation',
            self._obs_callback,
            10
        )

        self.reward_sub = self.create_subscription(
            Float32,
            '/current_reward',
            self._reward_callback,
            10
        )

        self.done_sub = self.create_subscription(
            Bool,
            '/episode_done',
            self._done_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_pub = self.create_publisher(Bool, '/request_reset', 10)
        
        # Debug: Check if we can publish
        print(f"Cmd_vel publisher created: {self.cmd_pub}")
        print(f"Reset publisher created: {self.reset_pub}")
        
        # Give time for other nodes to start
        print("Waiting 3 seconds for other nodes to start...")
        time.sleep(3.0)
        
        # Check topic connections
        self._check_topics()
        
        # Spin in separate thread
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self.spin_thread.start()
        
        print("ROS2 Gym environment initialized")

    # ===============================
    # Topic checking
    # ===============================
    def _check_topics(self):
        print("\nChecking topic connections...")
        
        # List all topics
        topic_info = self.get_topic_names_and_types()
        print(f"Total topics found: {len(topic_info)}")
        
        # Filter for relevant topics
        relevant = []
        for topic_name, topic_types in topic_info:
            if any(keyword in topic_name for keyword in 
                   ['observation', 'reward', 'done', 'cmd_vel', 'reset', 'odom', 'scan', 'goal']):
                relevant.append((topic_name, topic_types))
        
        print("\nRelevant topics:")
        for topic_name, topic_types in relevant:
            print(f"  - {topic_name} ({topic_types})")
            
        # Check if our topics exist
        required_topics = [
            '/normalized_observation',
            '/current_reward', 
            '/episode_done',
            '/cmd_vel',
            '/request_reset'
        ]
        
        print("\nChecking required topics:")
        for topic in required_topics:
            exists = any(topic in name for name, _ in topic_info)
            status = "✓" if exists else "✗"
            print(f"  {status} {topic}")
            
        return len(relevant) > 0

    # ===============================
    # Callbacks
    # ===============================
    def _obs_callback(self, msg):
        if len(msg.data) == 29:
            self.latest_obs = np.array(msg.data, dtype=np.float32)
            self.obs_event.set()
            # Debug: print first observation
            if not hasattr(self, 'first_obs_received'):
                self.first_obs_received = True
                print(f"✓ First observation received! Shape: {len(msg.data)}")
                print(f"  Sample values: {msg.data[:5]}...")
        else:
            print(f"✗ Wrong observation size: {len(msg.data)}")

    def _reward_callback(self, msg):
        self.latest_reward = float(msg.data)
        self.reward_event.set()

    def _done_callback(self, msg):
        self.latest_done = bool(msg.data)
        self.done_event.set()

    # ===============================
    # Helper: Wait for observation with timeout
    # ===============================
    def _wait_for_observation(self, timeout=3.0):
        """Wait for observation with timeout, returns observation or None"""
        start_time = time.time()
        
        # Clear the event first
        self.obs_event.clear()
        
        # Wait for event or timeout
        if self.obs_event.wait(timeout):
            obs = self.latest_obs.copy()
            return obs
        else:
            print(f"Timeout waiting for observation after {timeout}s")
            return None

    # ===============================
    # Gym API
    # ===============================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        print("\n" + "="*50)
        print("ENVIRONMENT RESET")
        print("="*50)
        
        # Clear all events
        self.obs_event.clear()
        self.reward_event.clear()
        self.done_event.clear()
        
        # Reset internal state
        self.current_step = 0
        self.latest_reward = 0.0
        self.latest_done = False
        
        # Send reset signal
        print("Sending reset signal...")
        reset_msg = Bool()
        reset_msg.data = True
        
        for i in range(3):  # Try multiple times
            self.reset_pub.publish(reset_msg)
            print(f"  Reset signal sent (attempt {i+1})")
            time.sleep(0.5)
        
        # Wait longer for initial observation
        print("\nWaiting for initial observation...")
        obs = self._wait_for_observation(timeout=5.0)
        
        if obs is None:
            print("✗ No observation received after reset!")
            print("Creating synthetic observation...")
            
            # Create a realistic observation
            obs = np.zeros(29, dtype=np.float32)
            # Lidar (20 values): mostly open space
            obs[0:20] = np.random.uniform(0.7, 1.0, 20)
            # Position (normalized)
            obs[20:22] = np.random.uniform(-0.5, 0.5, 2)
            # Orientation (sin/cos)
            obs[22:24] = [0.0, 1.0]  # Facing forward
            # Velocities (0)
            obs[24:26] = 0.0
            # Goal relative position
            obs[26:29] = np.random.uniform(-0.5, 0.5, 3)
            
            print("Using synthetic observation for this step")
        else:
            print(f"✓ Observation received! Shape: {obs.shape}")
            print(f"  Mean: {np.mean(obs):.3f}, Min: {np.min(obs):.3f}, Max: {np.max(obs):.3f}")
        
        print("="*50)
        return obs, {}

    def step(self, action):
        # Clip and convert action
        action = np.clip(action, -1.0, 1.0)
        
        print(f"\nStep {self.current_step + 1}:")
        print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}]")
        
        # Publish action as cmd_vel
        twist = Twist()
        twist.linear.x = float(action[0] * 0.22)  # Max 0.22 m/s
        twist.angular.z = float(action[1] * 2.84)  # Max 2.84 rad/s
        
        print(f"  Command: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f}")
        self.cmd_pub.publish(twist)
        
        # Wait for action to take effect
        wait_time = 0.2  # 200ms for simulation step
        print(f"  Waiting {wait_time}s for simulation...")
        time.sleep(wait_time)
        
        # Try to get observation
        print("  Waiting for observation...")
        obs = self._wait_for_observation(timeout=1.0)
        
        if obs is None:
            print("  ✗ No observation received!")
            # Create synthetic observation based on action
            obs = np.zeros(29, dtype=np.float32)
            # Simulate moving forward
            if abs(action[0]) > 0.1:
                obs[0:20] = np.random.uniform(0.5, 0.9, 20)  # Some obstacles
            else:
                obs[0:20] = np.random.uniform(0.7, 1.0, 20)  # Open space
            
            reward = -0.1  # Penalty for missing observation
        else:
            print(f"  ✓ Observation received")
            # Get reward if available
            if self.reward_event.is_set():
                reward = self.latest_reward
                self.reward_event.clear()
            else:
                reward = 0.0  # Default reward
        
        # Check done status
        done = self.latest_done if self.done_event.is_set() else False
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated = done
        truncated = (self.current_step >= self.max_steps)
        
        # Log summary
        print(f"  Step result: reward={reward:.3f}, done={terminated}, steps={self.current_step}")
        
        return obs, reward, terminated, truncated, {}

    def close(self):
        print("\nClosing environment...")
        self.destroy_node()


# ===============================
# TRAINING ENTRY POINT
# ===============================
def main():
    print("\n" + "="*60)
    print("MAZE NAVIGATION RL TRAINING")
    print("="*60)
    
    # Initialize ROS
    rclpy.init()
    
    # Create environment
    print("\nCreating environment...")
    env = ROS2RobotEnv(max_steps=200)
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    # Setup directories
    model_dir = "./models/ppo_maze"
    log_dir = "./tensorboard_logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\nDirectories:")
    print(f"  Models: {model_dir}")
    print(f"  Logs: {log_dir}")
    
    # Callback for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_dir,
        name_prefix="ppo_maze",
    )
    
    # Create the PPO model
    print("\nCreating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,  # Small for testing
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device='cpu',
    )
    
    print("✓ Model created successfully")
    
    # Train the model
    total_timesteps = 10000  # Small for testing
    
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=True,
            tb_log_name="ppo_maze_navigation",
            log_interval=1,
            progress_bar=False,
        )
        
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save the final model
        final_path = os.path.join(model_dir, "ppo_maze_final")
        model.save(final_path)
        print(f"\nModel saved to: {final_path}")
        
        # Cleanup
        env.close()
        rclpy.shutdown()
        
        print("\n" + "="*60)
        print("TRAINING SESSION COMPLETE")
        print("="*60)
        print(f"To view logs: tensorboard --logdir {log_dir}")
        print("="*60)


if __name__ == "__main__":
    main()