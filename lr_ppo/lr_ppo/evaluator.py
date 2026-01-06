# lr_ppo/lr_ppo/evaluator.py
#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
import time
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from .ppo_agent import PPOAgent
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import Odometry

class ROSIEvaluator(Node):
    def __init__(self, node_name='rosi_evaluator'):
        super().__init__(node_name)
        
        # Evaluation parameters
        self.declare_parameter('num_evaluation_episodes', 50)
        self.declare_parameter('max_steps_per_episode', 500)
        self.declare_parameter('model_path', './models/rosi_ppo_final.pth')
        self.declare_parameter('results_path', './evaluation_results/')
        self.declare_parameter('maze_name', 'maze1')
        
        # Load trained model
        model_path = self.get_parameter('model_path').value
        state_dim = 29
        action_dim = 2
        
        self.agent = PPOAgent(state_dim, action_dim, 'evaluator_agent')
        
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            self.get_logger().info(f"Loaded model from {model_path}")
        else:
            self.get_logger().error(f"Model not found: {model_path}")
            # Create a dummy model for testing
            self.get_logger().info("Using untrained model for evaluation")
        
        # Set to evaluation mode
        self.agent.actor.eval()
        
        # Evaluation metrics
        self.metrics = {
            'episode': [],
            'success': [],
            'collision': [],
            'timeout': [],
            'steps': [],
            'total_reward': [],
            'final_distance': []
        }
        
        # Create results directory
        results_path = self.get_parameter('results_path').value
        os.makedirs(results_path, exist_ok=True)
        
        # Publishers/Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_pub = self.create_publisher(Bool, '/request_reset', 10)
        
        self.create_subscription(Float32MultiArray, '/normalized_observation',
                               self.obs_callback, 10)
        self.create_subscription(Bool, '/collision',
                               self.collision_callback, 10)
        self.create_subscription(Bool, '/goal_reached',
                               self.success_callback, 10)
        self.create_subscription(Odometry, '/odom',
                               self.odom_callback, 10)
        
        # State
        self.current_obs = None
        self.collision = False
        self.success = False
        self.current_position = None
        self.goal_position = [3.5, 3.5]  # Default for maze1
        
        self.get_logger().info("ROSI Evaluator initialized")
    
    def obs_callback(self, msg):
        self.current_obs = np.array(msg.data, dtype=np.float32)
    
    def collision_callback(self, msg):
        self.collision = msg.data
    
    def success_callback(self, msg):
        self.success = msg.data
    
    def odom_callback(self, msg):
        self.current_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]
    
    def reset_environment(self):
        reset_msg = Bool()
        reset_msg.data = True
        self.reset_pub.publish(reset_msg)
        
        # Reset state
        self.current_obs = None
        self.collision = False
        self.success = False
        self.current_position = None
        
        time.sleep(2.0)
    
    def execute_action(self, action):
        twist_msg = Twist()
        
        max_linear = 0.22
        max_angular = 2.84
        
        linear_vel = action[0] * max_linear
        angular_vel = action[1] * max_angular
        
        twist_msg.linear.x = float(linear_vel)
        twist_msg.angular.z = float(angular_vel)
        
        self.cmd_vel_pub.publish(twist_msg)
    
    def compute_distance_to_goal(self):
        if self.current_position is None:
            return float('inf')
        
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        return np.sqrt(dx**2 + dy**2)
    
    def run_evaluation_episode(self, episode_num):
        self.reset_environment()
        
        episode_metrics = {
            'success': False,
            'collision': False,
            'timeout': False,
            'steps': 0,
            'total_reward': 0.0,
            'final_distance': 0.0
        }
        
        max_steps = self.get_parameter('max_steps_per_episode').value
        
        for step in range(max_steps):
            # Wait for observation
            if self.current_obs is None:
                time.sleep(0.01)
                continue
            
            # Get action (deterministic for evaluation)
            action, _, _ = self.agent.get_action(self.current_obs, deterministic=True)
            
            # Execute action
            self.execute_action(action)
            
            # Update metrics
            episode_metrics['steps'] += 1
            
            # Simple reward calculation for evaluation
            distance = self.compute_distance_to_goal()
            reward = -0.1  # Time penalty
            if distance < 0.3:  # Close to goal
                reward += 1.0
            
            episode_metrics['total_reward'] += reward
            
            # Wait for action to take effect
            time.sleep(0.1)
            
            # Check termination
            if self.success:
                episode_metrics['success'] = True
                episode_metrics['total_reward'] += 100.0  # Success bonus
                break
            
            if self.collision:
                episode_metrics['collision'] = True
                episode_metrics['total_reward'] -= 50.0  # Collision penalty
                break
        
        # Check for timeout
        if not episode_metrics['success'] and not episode_metrics['collision']:
            episode_metrics['timeout'] = True
        
        # Record final distance
        episode_metrics['final_distance'] = self.compute_distance_to_goal()
        
        # Log episode results
        self.get_logger().info(
            f"Episode {episode_num}: "
            f"Success={episode_metrics['success']}, "
            f"Steps={episode_metrics['steps']}, "
            f"Collision={episode_metrics['collision']}, "
            f"Distance={episode_metrics['final_distance']:.2f}m"
        )
        
        return episode_metrics
    
    def run_evaluation(self):
        num_episodes = self.get_parameter('num_evaluation_episodes').value
        maze_name = self.get_parameter('maze_name').value
        
        print("=" * 60)
        print(f"ROSI Evaluation - {maze_name}")
        print(f"Episodes: {num_episodes}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            metrics = self.run_evaluation_episode(episode + 1)
            
            # Store metrics
            self.metrics['episode'].append(episode + 1)
            self.metrics['success'].append(metrics['success'])
            self.metrics['collision'].append(metrics['collision'])
            self.metrics['timeout'].append(metrics['timeout'])
            self.metrics['steps'].append(metrics['steps'])
            self.metrics['total_reward'].append(metrics['total_reward'])
            self.metrics['final_distance'].append(metrics['final_distance'])
            
            # Wait between episodes
            time.sleep(1.0)
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
    
    def save_results(self):
        results_path = self.get_parameter('results_path').value
        maze_name = self.get_parameter('maze_name').value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Save CSV
        csv_file = os.path.join(results_path, f'evaluation_{maze_name}_{timestamp}.csv')
        df.to_csv(csv_file, index=False)
        
        # Calculate summary statistics
        summary = {
            'maze': maze_name,
            'total_episodes': len(df),
            'success_rate': df['success'].mean(),
            'collision_rate': df['collision'].mean(),
            'timeout_rate': df['timeout'].mean(),
            'avg_steps': df['steps'].mean(),
            'avg_reward': df['total_reward'].mean(),
            'avg_final_distance': df['final_distance'].mean(),
            'timestamp': timestamp
        }
        
        # Save summary
        summary_file = os.path.join(results_path, f'summary_{maze_name}_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create plots
        self.create_evaluation_plots(df, results_path, maze_name, timestamp)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for key, value in summary.items():
            if key not in ['timestamp', 'maze']:
                if 'rate' in key:
                    print(f"{key}: {value:.2%}")
                elif 'distance' in key:
                    print(f"{key}: {value:.2f}m")
                else:
                    print(f"{key}: {value:.2f}")
        
        # Check against expected results
        expected_rates = {
            'maze1': 0.80,
            'maze2': 0.60,
            'maze3': 0.40
        }
        
        if maze_name in expected_rates:
            if summary['success_rate'] >= expected_rates[maze_name]:
                print(f"\n✓ SUCCESS: Exceeded expected {expected_rates[maze_name]:.0%} for {maze_name}")
            else:
                print(f"\n⚠️  WARNING: Below expected {expected_rates[maze_name]:.0%} for {maze_name}")
        
        self.get_logger().info(f"Results saved to {results_path}")
    
    def create_evaluation_plots(self, df, save_path, maze_name, timestamp):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Success vs Episode
        axes[0, 0].plot(df['episode'], df['success'].cumsum() / (df.index + 1), 'g-', linewidth=2)
        axes[0, 0].axhline(y=df['success'].mean(), color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Success Rate')
        axes[0, 0].set_title(f'Success Rate Progress - {maze_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Length
        axes[0, 1].bar(df['episode'], df['steps'], alpha=0.6)
        axes[0, 1].axhline(y=df['steps'].mean(), color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title(f'Episode Length - {maze_name}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Final Distance Distribution
        axes[1, 0].hist(df['final_distance'], bins=20, alpha=0.7, color='purple')
        axes[1, 0].axvline(x=df['final_distance'].mean(), color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Final Distance to Goal (m)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Final Distance Distribution - {maze_name}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Termination Reasons
        termination_counts = {
            'Success': df['success'].sum(),
            'Collision': df['collision'].sum(),
            'Timeout': df['timeout'].sum()
        }
        
        axes[1, 1].pie(termination_counts.values(), labels=termination_counts.keys(),
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title(f'Termination Reasons - {maze_name}')
        
        plt.tight_layout()
        
        plot_file = os.path.join(save_path, f'evaluation_plots_{maze_name}_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

def main(args=None):
    rclpy.init(args=args)
    evaluator = ROSIEvaluator()
    
    try:
        evaluator.run_evaluation()
    except KeyboardInterrupt:
        evaluator.get_logger().info("Evaluation interrupted")
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()