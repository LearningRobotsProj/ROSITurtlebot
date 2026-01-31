#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import rclpy
from rclpy.node import Node
import os

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 128]):
        super().__init__()
        
        # Actor network
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mean_layer = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], action_dim)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        
        mean = torch.tanh(self.mean_layer(x))
        log_std = torch.tanh(self.log_std_layer(x)) * 0.5  # Constrain std
        
        # Ensure std is positive
        std = torch.exp(log_std) + 1e-4
        
        return mean, std

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=[256, 128]):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
        self.activation = nn.Tanh()
    
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        value = self.fc3(x)
        return value

class PPOMemory:
    def __init__(self, buffer_size=2048):
        self.buffer_size = buffer_size
        self.clear()
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def is_full(self):
        return len(self.states) >= self.buffer_size
    
    def __len__(self):
        return len(self.states)
    
    def get_tensors(self):
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        next_states = torch.FloatTensor(np.array(self.next_states))
        dones = torch.FloatTensor(np.array(self.dones))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        values = torch.FloatTensor(np.array(self.values))
        
        return states, actions, rewards, next_states, dones, log_probs, values

class PPOAgent(Node):
    def __init__(self, state_dim=29, action_dim=2, node_name='ppo_agent'):
        super().__init__(node_name)
        
        # PPO Parameters
        self.declare_parameter('learning_rate', 3e-4)
        self.declare_parameter('gamma', 0.99)
        self.declare_parameter('clip_epsilon', 0.2)
        self.declare_parameter('ppo_epochs', 10)
        self.declare_parameter('batch_size', 64)
        self.declare_parameter('entropy_coef', 0.01)
        self.declare_parameter('value_coef', 0.5)
        self.declare_parameter('max_grad_norm', 0.5)
        
        # Network dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Optimizers
        lr = self.get_parameter('learning_rate').value
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        
        # Memory
        self.memory = PPOMemory(buffer_size=2048)
        
        self.get_logger().info(f"PPO Agent initialized (state_dim={state_dim}, action_dim={action_dim})")
    
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            # Get action distribution
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.rsample()  # Use reparameterization trick
            
            # Clip action to [-1, 1]
            action = torch.clamp(action, -0.999, 0.999)
            
            # Get log probability
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Get value
            value = self.critic(state)
        
        return action.squeeze().numpy(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # Compute advantages in reverse
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # GAE
            advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return returns, advantages
    
    def update(self):
        """Update policy using PPO"""
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        # Get parameters
        clip_epsilon = self.get_parameter('clip_epsilon').value
        ppo_epochs = self.get_parameter('ppo_epochs').value
        batch_size = self.get_parameter('batch_size').value
        entropy_coef = self.get_parameter('entropy_coef').value
        value_coef = self.get_parameter('value_coef').value
        gamma = self.get_parameter('gamma').value
        
        # Get data from memory
        states, actions, rewards, next_states, dones, old_log_probs, old_values = self.memory.get_tensors()
        
        # Compute next value for the last state
        with torch.no_grad():
            if dones[-1]:
                next_value = 0.0
            else:
                next_value = self.critic(next_states[-1:]).item()
        
        # Convert to numpy for GAE computation
        rewards_np = rewards.numpy()
        values_np = old_values.numpy()
        dones_np = dones.numpy()
        
        # Compute GAE
        returns, advantages = self.compute_gae(
            rewards_np, values_np, next_value, dones_np, gamma, 0.95
        )
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # PPO Update
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        for epoch in range(ppo_epochs):
            # Create random indices
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                # Get batch
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current policy
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO Loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                values_pred = self.critic(batch_states).squeeze()
                critic_loss = nn.functional.mse_loss(values_pred, batch_returns)
                
                # Total Loss
                total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
                
                # Backpropagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                max_grad_norm = self.get_parameter('max_grad_norm').value
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                
                # Update
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Store losses
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy.item())
        
        # Clear memory
        self.memory.clear()
        
        # Compute average losses
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_entropy = np.mean(entropy_losses) if entropy_losses else 0.0
        
        self.get_logger().info(
            f"PPO Update: Actor Loss={avg_actor_loss:.4f}, "
            f"Critic Loss={avg_critic_loss:.4f}, "
            f"Entropy={avg_entropy:.4f}"
        )
        
        return avg_actor_loss, avg_critic_loss, avg_entropy
    
    def save_model(self, path):
        """Save model to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        
        self.get_logger().info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from file"""
        if not os.path.exists(path):
            self.get_logger().error(f"Model not found at {path}")
            return False
        
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.get_logger().info(f"Model loaded from {path}")
        return True

def main(args=None):
    rclpy.init(args=args)
    agent = PPOAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("PPO Agent stopped by user")
    except Exception as e:
        agent.get_logger().error(f"PPO Agent error: {str(e)}")
    finally:
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()