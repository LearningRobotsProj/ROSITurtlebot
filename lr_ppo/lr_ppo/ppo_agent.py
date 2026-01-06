# lr_ppo/lr_ppo/ppo_agent.py
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
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, state):
        features = self.network(state)
        mean = torch.tanh(self.mean_layer(features))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def get_batches(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.next_states)),
            torch.FloatTensor(np.array(self.dones)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.values))
        )

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
        
        # Network dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Optimizers
        lr = self.get_parameter('learning_rate').value
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
        self.get_logger().info(f"PPO Agent initialized (state_dim={state_dim}, action_dim={action_dim})")
    
    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            
            # Clip action to [-1, 1]
            action = torch.tanh(action)
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state)
        
        return action.squeeze().numpy(), log_prob.item(), value.item()
    
    def compute_returns(self, rewards, dones, values, next_value, gamma=0.99):
        returns = []
        advantages = []
        
        R = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - done)
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)
        advantages = returns - values
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self):
        if len(self.memory.states) == 0:
            return
        
        # Get data from memory
        states, actions, rewards, next_states, dones, old_log_probs, values = \
            self.memory.get_batches()
        
        # Compute next value
        with torch.no_grad():
            next_value = self.critic(torch.FloatTensor(next_states[-1:])).item()
        
        returns, advantages = self.compute_returns(
            rewards.numpy(), dones.numpy(), values.numpy(), next_value,
            self.get_parameter('gamma').value
        )
        
        # PPO update
        ppo_epochs = self.get_parameter('ppo_epochs').value
        batch_size = self.get_parameter('batch_size').value
        clip_epsilon = self.get_parameter('clip_epsilon').value
        
        for _ in range(ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Get current policy
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                
                # Losses
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                values_pred = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values_pred, batch_returns)
                
                # Total loss
                entropy_coef = self.get_parameter('entropy_coef').value
                total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
                
                # Backpropagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                # Update
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # Clear memory
        self.memory.clear()
        
        self.get_logger().info("PPO update completed")
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        self.get_logger().info(f"Model saved to {path}")
    
    def load_model(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"Model not found at {path}")
            return
        
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.get_logger().info(f"Model loaded from {path}")

def main(args=None):
    rclpy.init(args=args)
    agent = PPOAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

def debug_main():
    """Debug function to test PPO agent."""
    print("=== Testing PPO Agent ===")
    
    # Create agent
    agent = PPOAgent(state_dim=29, action_dim=2, node_name='test_agent')
    
    # Test 1: Action generation
    print("\n1. Testing action generation...")
    dummy_state = np.random.randn(29)
    action, log_prob, value = agent.get_action(dummy_state)
    
    if action.shape == (2,) and -1 <= action[0] <= 1 and -1 <= action[1] <= 1:
        print(f"✓ Action shape and range correct: [{action[0]:.3f}, {action[1]:.3f}]")
    else:
        print(f"✗ Action incorrect: shape={action.shape}, values={action}")
    
    # Test 2: Memory storage
    print("\n2. Testing memory storage...")
    for _ in range(5):
        state = np.random.randn(29)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state = np.random.randn(29)
        done = np.random.rand() > 0.5
        log_prob = np.random.randn()
        value = np.random.randn()
        
        agent.memory.store(state, action, reward, next_state, done, log_prob, value)
    
    states, actions, _, _, _, _, _ = agent.memory.get_batches()
    
    if len(states) == 5:
        print(f"✓ Memory stored {len(states)} transitions")
    else:
        print(f"✗ Memory storage failed: {len(states)} transitions")
    
    # Test 3: Model saving/loading
    print("\n3. Testing model saving/loading...")
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        # Save
        agent.save_model(temp_path)
        
        # Create new agent and load
        new_agent = PPOAgent(state_dim=29, action_dim=2, node_name='new_agent')
        new_agent.load_model(temp_path)
        
        # Compare actions
        test_state = np.random.randn(29)
        action1, _, _ = agent.get_action(test_state, deterministic=True)
        action2, _, _ = new_agent.get_action(test_state, deterministic=True)
        
        if np.allclose(action1, action2, rtol=1e-4):
            print("✓ Model save/load successful")
        else:
            print("✗ Model save/load failed")
            print(f"  Original: {action1}")
            print(f"  Loaded: {action2}")
    
    finally:
        # Cleanup
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return True

if __name__ == "__main__":
    main()