"""
Reinforcement Learning environment for Block Puzzle.
This module provides a Gym-compatible interface for the Block Puzzle game.
"""

import numpy as np
import gym
from gym import spaces
from .block_puzzle_game import BlockPuzzleGame
from .shapes import SHAPES

class BlockPuzzleEnv(gym.Env):
    """
    Block Puzzle environment for Reinforcement Learning.
    Follows the OpenAI Gym interface.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=8, max_steps=200, reward_shaping=True):
        """
        Initialize the environment.
        
        Args:
            grid_size: Size of the square grid (default: 8)
            max_steps: Maximum number of steps per episode
            reward_shaping: Whether to use shaped rewards
        """
        super(BlockPuzzleEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.steps = 0
        
        # Create the game
        self.game = BlockPuzzleGame(grid_size=grid_size)
        
        # Define action space
        # Action is (shape_idx, row, col)
        # For an 8x8 grid with 3 shapes, we have 3 * 8 * 8 = 192 possible actions
        self.action_space = spaces.MultiDiscrete([3, grid_size, grid_size])
        
        # Define observation space
        # Grid: 8x8 binary matrix
        # Shapes: 3 shapes, each with maximum size 5x5
        # We'll flatten these into a single array for the observation space
        grid_shape = (grid_size, grid_size)
        max_shape_size = 5
        shape_slots = 3
        
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=grid_shape, dtype=np.int8),
            'shapes': spaces.Box(low=0, high=1, 
                                shape=(shape_slots, max_shape_size, max_shape_size), 
                                dtype=np.int8),
            'valid_actions_mask': spaces.Box(low=0, high=1, 
                                          shape=(3, grid_size, grid_size), 
                                          dtype=np.int8)
        })
    
    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            dict: Initial observation
        """
        self.game = BlockPuzzleGame(grid_size=self.grid_size)
        self.steps = 0
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Tuple or array (shape_idx, row, col)
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Increment step counter
        self.steps += 1
        
        # Extract action components
        shape_idx, row, col = action
        
        # Take action in the game
        success, reward = self.game.place_shape(shape_idx, row, col)
        
        # Check if the action was valid
        if not success:
            # Invalid action penalty
            if self.reward_shaping:
                reward = -5.0
            else:
                reward = 0.0
        
        # Get the next observation
        observation = self._get_observation()
        
        # Check if the episode is done
        done = self.game.game_over or self.steps >= self.max_steps
        
        # Additional info
        info = {
            'score': self.game.score,
            'steps': self.steps,
            'rows_cleared': self.game.total_rows_cleared,
            'cols_cleared': self.game.total_cols_cleared,
            'invalid_action': not success,
            'game_over': self.game.game_over
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            numpy array if mode is 'rgb_array', None otherwise
        """
        if mode == 'human':
            # Print the grid
            print("\nGrid:")
            for row in self.game.grid:
                print(" ".join(["■" if cell == 1 else "□" for cell in row]))
            
            # Print available shapes
            print("\nAvailable shapes:")
            for i, shape in enumerate(self.game.current_shapes):
                print(f"Shape {i} ({self.game.current_shape_names[i]}):")
                for row in shape:
                    print(" ".join(["■" if cell == 1 else "□" for cell in row]))
            
            # Print score
            print(f"\nScore: {self.game.score}")
            print(f"Rows cleared: {self.game.total_rows_cleared}")
            print(f"Columns cleared: {self.game.total_cols_cleared}")
            print(f"Game over: {self.game.game_over}")
            
            return None
        
        elif mode == 'rgb_array':
            # For simplicity, return a basic visualization
            # In a real implementation, this would return a proper image
            # that matches the pygame visualization
            grid_viz = np.zeros((self.grid_size * 10, self.grid_size * 10, 3), dtype=np.uint8)
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    color = [0, 0, 139] if self.game.grid[i, j] == 1 else [255, 255, 255]
                    grid_viz[i*10:(i+1)*10, j*10:(j+1)*10] = color
            
            return grid_viz
    
    def _get_observation(self):
        """
        Convert the game state to an observation.
        
        Returns:
            dict: Observation
        """
        # Get the current grid
        grid = self.game.grid.copy()
        
        # Prepare shape observations with padding
        max_shape_size = 5
        shape_slots = 3
        shapes = np.zeros((shape_slots, max_shape_size, max_shape_size), dtype=np.int8)
        
        for i, shape in enumerate(self.game.current_shapes):
            if i >= shape_slots:
                break
                
            shape_height, shape_width = shape.shape
            shapes[i, :shape_height, :shape_width] = shape
        
        # Create a valid actions mask
        valid_actions_mask = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int8)
        
        for shape_idx, row, col in self.game.get_valid_moves():
            if shape_idx < 3:  # Ensure we stay within bounds
                valid_actions_mask[shape_idx, row, col] = 1
        
        return {
            'grid': grid,
            'shapes': shapes,
            'valid_actions_mask': valid_actions_mask
        }
    
    def get_valid_actions(self):
        """
        Get all valid actions.
        
        Returns:
            list: List of valid actions as (shape_idx, row, col) tuples
        """
        return self.game.get_valid_moves()
    
    def get_action_mask(self):
        """
        Get a binary mask of valid actions.
        
        Returns:
            numpy array: Binary mask of shape (3, grid_size, grid_size)
        """
        mask = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int8)
        
        for shape_idx, row, col in self.game.get_valid_moves():
            if shape_idx < 3:  # Ensure we stay within bounds
                mask[shape_idx, row, col] = 1
        
        return mask

    def sample_valid_action(self):
        """
        Sample a random valid action.
        
        Returns:
            tuple: (shape_idx, row, col) or None if no valid actions
        """
        valid_actions = self.game.get_valid_moves()
        if not valid_actions:
            return None
        
        return valid_actions[np.random.randint(0, len(valid_actions))]
