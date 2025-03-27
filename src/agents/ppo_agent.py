"""
Proximal Policy Optimization (PPO) agent for Block Puzzle.
This implementation uses PyTorch and supports action masking.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_agent import BaseAgent
from collections import deque

class MaskedCategorical(Categorical):
    """
    Extension of PyTorch's Categorical distribution that supports masking.
    This prevents sampling of invalid actions by setting their probabilities to zero.
    """
    
    def __init__(self, probs=None, logits=None, mask=None):
        """
        Initialize the distribution.
        
        Args:
            probs: Action probabilities
            logits: Log probabilities
            mask: Binary mask for valid actions (1 for valid, 0 for invalid)
        """
        if mask is None:
            super(MaskedCategorical, self).__init__(probs=probs, logits=logits)
        else:
            if logits is not None:
                # Apply mask by setting logits for invalid actions to a large negative value
                logits = logits.clone()
                # Fix: Ensure mask has the same shape as logits
                if logits.dim() > mask.dim():
                    # Add batch dimension to mask if needed
                    mask = mask.unsqueeze(0)
                # Now apply the mask
                logits[mask == 0] = -1e9
                super(MaskedCategorical, self).__init__(logits=logits)
            elif probs is not None:
                # Apply mask by setting probabilities for invalid actions to 0
                probs = probs.clone()
                # Fix: Ensure mask has the same shape as probs
                if probs.dim() > mask.dim():
                    mask = mask.unsqueeze(0)
                probs[mask == 0] = 0
                # Renormalize probabilities
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)  # Added small epsilon to prevent division by zero
                super(MaskedCategorical, self).__init__(probs=probs)
            else:
                raise ValueError("Either probs or logits must be provided")

class PPONetwork(nn.Module):
    """
    Neural network for PPO agent.
    """
    
    def __init__(self, obs_shape, action_shape, hidden_dim=128):
        """Initialize the network."""
        super(PPONetwork, self).__init__()
        
        # Flatten input dimensions
        grid_flat = int(np.prod(obs_shape['grid']))
        shapes_flat = int(np.prod(obs_shape['shapes']))
        
        # Feature extractor for grid
        self.grid_encoder = nn.Sequential(
            nn.Linear(grid_flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Feature extractor for shapes
        self.shapes_encoder = nn.Sequential(
            nn.Linear(shapes_flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined features
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (outputs logits for flattened action space)
        action_dim = int(np.prod(action_shape))
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs):
        """Forward pass through the network."""
        # Extract observations
        grid = obs['grid'].reshape(obs['grid'].size(0), -1)
        shapes = obs['shapes'].reshape(obs['shapes'].size(0), -1)
        
        # Process grid and shapes
        grid_features = self.grid_encoder(grid)
        shapes_features = self.shapes_encoder(shapes)
        
        # Combine features
        combined = torch.cat([grid_features, shapes_features], dim=1)
        features = self.combined_encoder(combined)
        
        # Get policy and value
        action_logits = self.policy_head(features)
        state_value = self.value_head(features)
        
        return action_logits, state_value

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent for Block Puzzle.
    """
        
    def __init__(self, env, hidden_dim=128, lr=3e-4, gamma=0.99, 
                 clip_ratio=0.2, target_kl=0.01, gae_lambda=0.95, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, 
                 device=None):
        """Initialize the PPO agent."""
        super(PPOAgent, self).__init__(env)
        
        # Set hyperparameters
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.gae_lambda = gae_lambda  # Add this line
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Rest of the method stays the same...
        
        # Set device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create network
        obs_shape = {
            'grid': env.observation_space['grid'].shape,
            'shapes': env.observation_space['shapes'].shape
        }
        
        action_shape = env.action_space.nvec
        
        self.network = PPONetwork(obs_shape, action_shape, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Initialize buffers
        self.reset_buffers()
        
        # Initialize action converter
        self.grid_size = env.grid_size
        
        # Metrics
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
        }
    
    def reset_buffers(self):
        """Reset experience buffers."""
        self.states = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.next_states = []
    
    def preprocess_observation(self, observation):
        """Preprocess observation for neural network input."""
        # Convert numpy arrays to PyTorch tensors
        grid = torch.FloatTensor(observation['grid']).unsqueeze(0).to(self.device)
        shapes = torch.FloatTensor(observation['shapes']).unsqueeze(0).to(self.device)
        mask = torch.FloatTensor(observation['valid_actions_mask']).reshape(1, -1).to(self.device)
        
        return {
            'grid': grid,
            'shapes': shapes,
            'mask': mask
        }
    
    def act(self, observation, reward=None, done=None):
        """Select an action based on the current observation."""
        # Store previous step information if available
        if reward is not None and done is not None and self.states:
            self.rewards.append(reward)
            self.dones.append(done)
            self.next_states.append(observation)
        
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation)
        
        # Store the state
        self.states.append(observation)
        
        # Get action mask
        action_mask = processed_obs['mask']
        
        # Set network to evaluation mode for inference
        self.network.eval()
        
        with torch.no_grad():
            # Forward pass
            action_logits, value = self.network({
                'grid': processed_obs['grid'], 
                'shapes': processed_obs['shapes']
            })
            
            # Apply mask by setting logits for invalid actions to -inf
            masked_logits = action_logits.clone()
            masked_logits[action_mask == 0] = -1e9
            
            # Create distribution and sample action
            dist = Categorical(logits=masked_logits)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)
            
            # Convert action index to 3D action (shape_idx, row, col)
            action_flat = action_index.item()
            
            # Convert linear index to 3D action based on environment's action space
            shape_idx = action_flat // (self.grid_size * self.grid_size)
            remainder = action_flat % (self.grid_size * self.grid_size)
            row = remainder // self.grid_size
            col = remainder % self.grid_size
            
            action = (shape_idx, row, col)
        
        # Store experience for learning
        if self.training:
            self.actions.append(action_flat)
            self.action_masks.append(action_mask.cpu().numpy())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        return action
    
    def update(self):
        """Update policy based on collected experience."""
        # Ensure we have enough data
        if len(self.states) <= 1:
            print("Not enough data to update policy")
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        # Process all rewards and values
        returns = []
        advantages = []
        
        # Check if we need to add the final reward - only print warning once per update
        if len(self.rewards) < len(self.states) - 1:
            # Add a zero reward as placeholder
            self.rewards.append(0)
            self.dones.append(False)
            self.next_states.append(self.states[-1])
        
        # Calculate returns and advantages
        next_value = 0
        next_advantage = 0
        next_return = 0  # Initialize next_return
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                next_return = 0
                next_advantage = 0
                next_value = 0
            else:
                next_state = self.next_states[i]
                processed_next = self.preprocess_observation(next_state)
                
                with torch.no_grad():
                    _, next_value = self.network({
                        'grid': processed_next['grid'],
                        'shapes': processed_next['shapes']
                    })
                    next_value = next_value.item()
            
            # Calculate return (discounted sum of rewards)
            returns.insert(0, self.rewards[i] + self.gamma * next_return * (1 - self.dones[i]))
            
            # Calculate advantage (TD error + discounted advantage)
            td_error = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            advantages.insert(0, td_error + self.gamma * self.gae_lambda * next_advantage * (1 - self.dones[i]))
            
            # Update values for next iteration
            next_return = returns[0]
            next_advantage = advantages[0]
            next_value = self.values[i]
        
        # Convert data to tensors
        observations = self.states[:len(returns)]
        processed_obs = [self.preprocess_observation(obs) for obs in observations]
        
        obs_grid = torch.cat([obs['grid'] for obs in processed_obs])
        obs_shapes = torch.cat([obs['shapes'] for obs in processed_obs])
        actions = torch.LongTensor(self.actions[:len(returns)]).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs[:len(returns)]).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        self.network.train()
        
        # Forward pass
        logits, values = self.network({'grid': obs_grid, 'shapes': obs_shapes})
        
        # Calculate policy loss
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate ratios and clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_tensor
        policy_loss = -torch.min(ratio * advantages_tensor, clip_adv).mean()
        
        # Fix the tensor shape issue
        values = values.view(-1)  # Reshape to match returns_tensor
        value_loss = F.mse_loss(values, returns_tensor)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Perform update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Reset buffers
        self.reset_buffers()
        
        # Update metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        return metrics

    def learn(self, observation, action, reward, next_observation, done):
        """Update agent's policy - not used directly in PPO."""
        pass
    
    def save(self, filepath):
        """Save the agent's model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, filepath)
    
    def load(self, filepath):
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.metrics = checkpoint['metrics']