"""
Core game logic for Block Puzzle, separated from rendering.
This module contains the game mechanics that are shared between
the RL environment and the visualization.
"""

import numpy as np
import random
from .shapes import get_random_shapes, SHAPES

class BlockPuzzleGame:
    """
    Core game logic for Block Puzzle.
    This class implements the game mechanics without any rendering.
    """
    def __init__(self, grid_size=8, num_shapes=3):
        """
        Initialize a new game.
        
        Args:
            grid_size: Size of the square grid (default: 8)
            num_shapes: Number of shapes available at a time (default: 3)
        """
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.score = 0
        self.consecutive_clears = 0
        self.current_shapes = []
        self.current_shape_names = []
        self.game_over = False
        self.total_rows_cleared = 0
        self.total_cols_cleared = 0

        # Reward shaping parameters
        self.block_reward = 1.0               # Reward per block placed
        self.base_line_clear_rewards = {1: 10, 2: 25}  # Bonus for clearing 1 or 2 lines
        self.multi_line_bonus = 50            # Base bonus for clearing 3 or more lines
        self.additional_line_bonus = 10       # Additional bonus per extra line beyond 3
        self.streak_bonus_factor = 5          # Extra bonus for consecutive clears
        self.gap_penalty_coef = 0.5           # Penalty per isolated gap on the board
        self.game_over_penalty = 50.0         # Penalty applied when no valid moves remain

        # Draw initial shapes
        self.draw_new_shapes()

    def draw_new_shapes(self, shape_categories=None):
        """
        Draw new random shapes if there are none available.
        
        Args:
            shape_categories: Optional list of shape categories to draw from
        """
        if not self.current_shapes:
            self.current_shapes, self.current_shape_names = get_random_shapes(
                n=self.num_shapes, categories=shape_categories
            )

    def can_place_shape(self, shape, row, col):
        """
        Check if a shape can be placed at the given position.
        
        Args:
            shape: The shape to place (2D numpy array)
            row: Row index
            col: Column index
            
        Returns:
            bool: True if the shape can be placed, False otherwise
        """
        shape_height, shape_width = shape.shape

        # Check if the shape is within bounds
        if (row < 0 or col < 0 or 
            row + shape_height > self.grid_size or 
            col + shape_width > self.grid_size):
            return False

        # Check if the shape overlaps with existing blocks
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1 and self.grid[row + i, col + j] == 1:
                    return False

        return True

    def place_shape(self, shape_idx, row, col):
        """
        Place the specified shape at the given position.
        
        Args:
            shape_idx: Index of the shape in current_shapes
            row: Row index
            col: Column index
            
        Returns:
            bool: True if the shape was placed, False otherwise
            float: Reward for the action
        """
        # Validate inputs
        if (shape_idx >= len(self.current_shapes) or 
            self.game_over or 
            not isinstance(shape_idx, int) or 
            not isinstance(row, int) or 
            not isinstance(col, int)):
            return False, 0.0

        shape = self.current_shapes[shape_idx]
        if not self.can_place_shape(shape, row, col):
            return False, 0.0

        # Place the shape
        shape_height, shape_width = shape.shape
        blocks_placed = 0
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1:
                    self.grid[row + i, col + j] = 1
                    blocks_placed += 1

        # Compute reward components
        reward = blocks_placed * self.block_reward

        # Clear completed rows and columns and compute bonus
        rows_cleared, cols_cleared = self.clear_lines()
        self.total_rows_cleared += rows_cleared
        self.total_cols_cleared += cols_cleared
        total_cleared = rows_cleared + cols_cleared

        if total_cleared > 0:
            self.consecutive_clears += 1
            streak_bonus = (self.consecutive_clears - 1) * self.streak_bonus_factor if self.consecutive_clears > 1 else 0
            
            if total_cleared in self.base_line_clear_rewards:
                reward += self.base_line_clear_rewards[total_cleared] + streak_bonus
            else:  # For 3 or more lines cleared
                extra_lines = total_cleared - 3
                reward += self.multi_line_bonus + extra_lines * self.additional_line_bonus + streak_bonus
        else:
            self.consecutive_clears = 0

        # Penalize for empty gaps that are hard to fill
        gaps = self.count_empty_gaps()
        reward -= gaps * self.gap_penalty_coef

        # Update cumulative score with computed reward
        self.score += reward

        # Remove the used shape
        self.current_shapes.pop(shape_idx)
        self.current_shape_names.pop(shape_idx)

        # If no shapes remain, draw new ones and check for game over
        if not self.current_shapes:
            self.draw_new_shapes()
            if self.is_game_over():
                self.game_over = True
                reward -= self.game_over_penalty
                self.score -= self.game_over_penalty

        return True, reward

    def clear_lines(self):
        """
        Clear fully filled rows and columns.
        
        Returns:
            tuple: (rows_cleared, cols_cleared)
        """
        rows_cleared = 0
        cols_cleared = 0

        # Clear full rows
        for i in range(self.grid_size):
            if np.all(self.grid[i, :] == 1):
                self.grid[i, :] = 0
                rows_cleared += 1

        # Clear full columns
        for j in range(self.grid_size):
            if np.all(self.grid[:, j] == 1):
                self.grid[:, j] = 0
                cols_cleared += 1

        return rows_cleared, cols_cleared

    def is_game_over(self):
        """Check if no valid moves exist."""
        for shape_idx in range(len(self.current_shapes)):
            shape = self.current_shapes[shape_idx]
            shape_height, shape_width = shape.shape
            
            # Check all possible positions
            for row in range(self.grid_size - shape_height + 1):
                for col in range(self.grid_size - shape_width + 1):
                    if self.can_place_shape(shape, row, col):
                        return False
        
        return True

    def get_valid_moves(self, shape_idx=None):
        """
        Return all valid (shape_idx, row, col) placements or for a specific shape.
        
        Args:
            shape_idx: Optional index of the shape to check
            
        Returns:
            list: List of valid moves as (shape_idx, row, col) tuples
        """
        valid_moves = []
        
        # If shape_idx is specified, only check that shape
        if shape_idx is not None:
            if shape_idx >= len(self.current_shapes):
                return valid_moves
                
            shape = self.current_shapes[shape_idx]
            shape_height, shape_width = shape.shape
            
            for row in range(0, self.grid_size - shape_height + 1):
                for col in range(0, self.grid_size - shape_width + 1):
                    if self.can_place_shape(shape, row, col):
                        valid_moves.append((shape_idx, row, col))
            
            return valid_moves
        
        # Otherwise check all shapes
        for shape_idx, shape in enumerate(self.current_shapes):
            shape_height, shape_width = shape.shape
            
            for row in range(0, self.grid_size - shape_height + 1):
                for col in range(0, self.grid_size - shape_width + 1):
                    if self.can_place_shape(shape, row, col):
                        valid_moves.append((shape_idx, row, col))
        
        return valid_moves

    def count_empty_gaps(self):
        """
        Count isolated empty cells that are surrounded by filled cells or borders.
        
        Returns:
            int: Number of empty gaps
        """
        gaps = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:
                    surrounded = 0
                    # Check top
                    if i == 0 or self.grid[i - 1, j] == 1:
                        surrounded += 1
                    # Check bottom
                    if i == self.grid_size - 1 or self.grid[i + 1, j] == 1:
                        surrounded += 1
                    # Check left
                    if j == 0 or self.grid[i, j - 1] == 1:
                        surrounded += 1
                    # Check right
                    if j == self.grid_size - 1 or self.grid[i, j + 1] == 1:
                        surrounded += 1
                    
                    # If surrounded on 3 or more sides, count as a gap
                    if surrounded >= 3:
                        gaps += 1
        return gaps

    def get_state_representation(self):
        """
        Get a representation of the current game state.
        
        Returns:
            dict: State representation with different components
        """
        # Basic state components
        state = {
            'grid': self.grid.copy(),
            'shapes': [shape.copy() for shape in self.current_shapes],
            'shape_names': self.current_shape_names.copy(),
            'score': self.score,
            'consecutive_clears': self.consecutive_clears,
            'game_over': self.game_over,
            'total_rows_cleared': self.total_rows_cleared,
            'total_cols_cleared': self.total_cols_cleared,
        }
        
        # Advanced state features
        # Row completeness (how many cells are filled in each row)
        row_completeness = np.sum(self.grid, axis=1) / self.grid_size
        # Column completeness
        col_completeness = np.sum(self.grid, axis=0) / self.grid_size
        # Pattern potential (how close rows/cols are to being completed)
        pattern_potential = np.concatenate([row_completeness, col_completeness])
        # Total board fill percentage
        fill_percentage = np.sum(self.grid) / (self.grid_size * self.grid_size)
        
        state['pattern_potential'] = pattern_potential
        state['fill_percentage'] = fill_percentage
        
        return state

    def reset(self):
        """
        Reset the game to its initial state.
        
        Returns:
            dict: Initial state representation
        """
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.consecutive_clears = 0
        self.current_shapes = []
        self.current_shape_names = []
        self.game_over = False
        self.total_rows_cleared = 0
        self.total_cols_cleared = 0
        
        self.draw_new_shapes()
        
        return self.get_state_representation()

    def clone(self):
        """
        Return a deep copy of the current game state.
        
        Returns:
            BlockPuzzleGame: A clone of the current game
        """
        clone = BlockPuzzleGame(grid_size=self.grid_size, num_shapes=self.num_shapes)
        clone.grid = self.grid.copy()
        clone.score = self.score
        clone.consecutive_clears = self.consecutive_clears
        clone.current_shapes = [shape.copy() for shape in self.current_shapes]
        clone.current_shape_names = self.current_shape_names.copy()
        clone.game_over = self.game_over
        clone.total_rows_cleared = self.total_rows_cleared
        clone.total_cols_cleared = self.total_cols_cleared
        
        return clone
