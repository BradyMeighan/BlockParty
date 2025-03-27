"""
Base agent class for Block Puzzle reinforcement learning.
"""

import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for Block Puzzle agents.
    All agent implementations should inherit from this class.
    """
    
    def __init__(self, env):
        """
        Initialize the agent.
        
        Args:
            env: The BlockPuzzleEnv environment
        """
        self.env = env
        self.training = True
    
    @abstractmethod
    def act(self, observation, reward=None, done=None):
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current environment observation
            reward: Optional reward from the previous step
            done: Optional done flag from the previous step
            
        Returns:
            tuple: Selected action as (shape_idx, row, col)
        """
        pass
    
    @abstractmethod
    def learn(self, observation, action, reward, next_observation, done):
        """
        Update the agent's policy based on experience.
        
        Args:
            observation: Current observation
            action: Taken action
            reward: Received reward
            next_observation: Next observation
            done: Whether the episode is done
        """
        pass
    
    def train(self):
        """Set the agent to training mode."""
        self.training = True
    
    def eval(self):
        """Set the agent to evaluation mode."""
        self.training = False
    
    def save(self, filepath):
        """
        Save the agent's model to a file.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    def load(self, filepath):
        """
        Load the agent's model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        pass


class RandomAgent(BaseAgent):
    """
    Random agent that selects actions uniformly at random from valid actions.
    This serves as a baseline for comparison.
    """
    
    def __init__(self, env):
        """
        Initialize the random agent.
        
        Args:
            env: The BlockPuzzleEnv environment
        """
        super(RandomAgent, self).__init__(env)
    
    def act(self, observation, reward=None, done=None):
        """
        Select a random valid action.
        
        Args:
            observation: Current environment observation
            reward: Ignored
            done: Ignored
            
        Returns:
            tuple: Selected action as (shape_idx, row, col)
        """
        valid_action = self.env.sample_valid_action()
        if valid_action is None:
            # If no valid actions, return a random action
            # This should never happen in normal gameplay
            return (0, 0, 0)
        return valid_action
    
    def learn(self, observation, action, reward, next_observation, done):
        """
        Random agent doesn't learn.
        
        Args:
            observation: Ignored
            action: Ignored
            reward: Ignored
            next_observation: Ignored
            done: Ignored
        """
        pass


class HeuristicAgent(BaseAgent):
    """
    Heuristic agent that uses simple rules to select actions.
    This serves as an intermediate baseline between random and learned policies.
    """
    
    def __init__(self, env):
        """
        Initialize the heuristic agent.
        
        Args:
            env: The BlockPuzzleEnv environment
        """
        super(HeuristicAgent, self).__init__(env)
    
    def act(self, observation, reward=None, done=None):
        """
        Select an action based on heuristics.
        
        Heuristics:
        1. Prefer moves that fill rows/columns that are almost complete
        2. Avoid creating isolated gaps
        3. Prefer placing larger shapes over smaller ones
        
        Args:
            observation: Current environment observation
            reward: Ignored
            done: Ignored
            
        Returns:
            tuple: Selected action as (shape_idx, row, col)
        """
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return (0, 0, 0)  # Should never happen in normal gameplay
        
        grid = observation['grid']
        shapes = observation['shapes']
        
        # Calculate row and column fill counts
        row_fill = np.sum(grid, axis=1)
        col_fill = np.sum(grid, axis=0)
        
        best_action = None
        best_score = float('-inf')
        
        for shape_idx, row, col in valid_actions:
            shape = shapes[shape_idx]
            shape_height, shape_width = 0, 0
            
            # Find actual shape dimensions by removing zero padding
            for i in range(shape.shape[0]):
                if np.any(shape[i]):
                    shape_height += 1
            
            for j in range(shape.shape[1]):
                if np.any(shape[:, j]):
                    shape_width += 1
            
            # Skip if shape dimensions are 0 (should never happen)
            if shape_height == 0 or shape_width == 0:
                continue
            
            # Create a copy of the game state to simulate the move
            game_copy = self.env.game.clone()
            game_copy.place_shape(shape_idx, row, col)
            
            # Score based on:
            # 1. Number of blocks placed
            blocks_placed = np.sum(shape[:shape_height, :shape_width])
            
            # 2. Number of rows/columns that would be filled
            new_row_fill = np.sum(game_copy.grid, axis=1)
            new_col_fill = np.sum(game_copy.grid, axis=0)
            
            rows_filled = np.sum(new_row_fill == self.env.grid_size) - np.sum(row_fill == self.env.grid_size)
            cols_filled = np.sum(new_col_fill == self.env.grid_size) - np.sum(col_fill == self.env.grid_size)
            
            # 3. Progress towards filling rows/columns
            row_progress = np.sum((new_row_fill - row_fill) * (row_fill / self.env.grid_size))
            col_progress = np.sum((new_col_fill - col_fill) * (col_fill / self.env.grid_size))
            
            # 4. Number of isolated gaps created
            gaps_before = self.env.game.count_empty_gaps()
            gaps_after = game_copy.count_empty_gaps()
            gaps_created = max(0, gaps_after - gaps_before)
            
            # Combine scores with weights
            score = (
                blocks_placed * 1.0 +
                (rows_filled + cols_filled) * 10.0 +
                (row_progress + col_progress) * 0.5 -
                gaps_created * 2.0
            )
            
            if score > best_score:
                best_score = score
                best_action = (shape_idx, row, col)
        
        return best_action
    
    def learn(self, observation, action, reward, next_observation, done):
        """
        Heuristic agent doesn't learn.
        
        Args:
            observation: Ignored
            action: Ignored
            reward: Ignored
            next_observation: Ignored
            done: Ignored
        """
        pass
