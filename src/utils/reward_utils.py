"""
Utility functions for reward calculation and shaping.
"""

import numpy as np
from .state_utils import (
    count_isolated_empty_cells,
    calculate_connectivity,
    calculate_corner_edge_usage,
    calculate_clearance_potential,
    calculate_evenness,
    pattern_potential,
    calculate_open_lines
)

class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self):
        pass
    
    def calculate_reward(self, prev_state, action, next_state, game_info):
        """
        Calculate reward for a state transition.
        
        Args:
            prev_state: Previous game state
            action: Action taken
            next_state: Resulting game state
            game_info: Additional game information
            
        Returns:
            float: Calculated reward
        """
        raise NotImplementedError("Subclasses must implement calculate_reward")

class BasicReward(RewardFunction):
    """
    Basic reward function that gives:
    - Points for placing blocks
    - Bonus for clearing lines
    - Penalty for game over
    """
    
    def __init__(self, block_reward=1.0, line_clear_reward=10.0, game_over_penalty=50.0):
        """
        Initialize the basic reward function.
        
        Args:
            block_reward: Reward per block placed
            line_clear_reward: Reward per line cleared
            game_over_penalty: Penalty for game over
        """
        super().__init__()
        self.block_reward = block_reward
        self.line_clear_reward = line_clear_reward
        self.game_over_penalty = game_over_penalty
    
    def calculate_reward(self, prev_state, action, next_state, game_info):
        """
        Calculate reward for a state transition.
        
        Args:
            prev_state: Previous game state (ignored)
            action: Action taken (shape_idx, row, col)
            next_state: Resulting game state (ignored)
            game_info: Dictionary with:
                - blocks_placed: Number of blocks placed
                - lines_cleared: Number of lines cleared
                - game_over: Whether the game is over
            
        Returns:
            float: Calculated reward
        """
        reward = 0.0
        
        # Reward for placing blocks
        reward += game_info['blocks_placed'] * self.block_reward
        
        # Bonus for clearing lines
        reward += game_info['lines_cleared'] * self.line_clear_reward
        
        # Penalty for game over
        if game_info['game_over']:
            reward -= self.game_over_penalty
        
        return reward

class AdvancedReward(RewardFunction):
    """
    Advanced reward function that considers various aspects of the game state.
    Includes reward shaping to guide the agent toward better strategies.
    """
    
    def __init__(self):
        """Initialize the advanced reward function with default parameters."""
        super().__init__()
        
        # Reward components
        self.block_reward = 1.0                 # Reward per block placed
        self.base_line_clear_rewards = {        # Bonuses for clearing 1 or 2 lines
            1: 10.0,
            2: 25.0
        }
        self.multi_line_bonus = 50.0            # Base bonus for clearing 3+ lines
        self.additional_line_bonus = 10.0       # Bonus per extra line beyond 3
        self.streak_bonus_factor = 5.0          # Bonus multiplier for consecutive clears
        
        # Penalty components
        self.gap_penalty_coef = 0.5             # Penalty per isolated gap
        self.game_over_penalty = 50.0           # Penalty for game over
        
        # Shaping components
        self.connectivity_bonus = 0.3           # Bonus for improving connectivity
        self.pattern_bonus = 0.3                # Bonus for improving pattern potential
        self.open_line_bonus = 0.2              # Bonus for preserving open lines
        self.evenness_bonus = 0.2               # Bonus for improving evenness
        self.corner_edge_bonus = 0.2            # Bonus for using corners/edges
    
    def calculate_reward(self, prev_state, action, next_state, game_info):
        """
        Calculate reward for a state transition with advanced shaping.
        
        Args:
            prev_state: Dictionary with:
                - grid: Previous grid state
            action: Action taken (shape_idx, row, col)
            next_state: Dictionary with:
                - grid: New grid state
            game_info: Dictionary with:
                - blocks_placed: Number of blocks placed
                - rows_cleared: Number of rows cleared
                - cols_cleared: Number of columns cleared
                - consecutive_clears: Number of consecutive clears
                - game_over: Whether the game is over
            
        Returns:
            float: Calculated reward
        """
        reward = 0.0
        
        # Reward for placing blocks
        blocks_placed = game_info['blocks_placed']
        reward += blocks_placed * self.block_reward
        
        # Extract line clear information
        rows_cleared = game_info['rows_cleared']
        cols_cleared = game_info['cols_cleared']
        total_cleared = rows_cleared + cols_cleared
        consecutive_clears = game_info['consecutive_clears']
        
        # Bonus for clearing lines
        if total_cleared > 0:
            streak_bonus = (consecutive_clears - 1) * self.streak_bonus_factor if consecutive_clears > 1 else 0
            
            if total_cleared in self.base_line_clear_rewards:
                reward += self.base_line_clear_rewards[total_cleared] + streak_bonus
            else:  # For 3 or more lines cleared
                extra_lines = total_cleared - 3
                reward += self.multi_line_bonus + extra_lines * self.additional_line_bonus + streak_bonus
        
        # Penalty for isolated gaps
        if 'grid' in next_state:
            next_gaps = count_isolated_empty_cells(next_state['grid'])
            reward -= next_gaps * self.gap_penalty_coef
        
        # Penalty for game over
        if game_info['game_over']:
            reward -= self.game_over_penalty
        
        # Reward shaping based on state transitions
        if 'grid' in prev_state and 'grid' in next_state:
            prev_grid = prev_state['grid']
            next_grid = next_state['grid']
            
            # Connectivity improvement
            prev_connectivity = calculate_connectivity(prev_grid)
            next_connectivity = calculate_connectivity(next_grid)
            connectivity_change = next_connectivity - prev_connectivity
            reward += connectivity_change * self.connectivity_bonus
            
            # Pattern potential improvement
            prev_pattern = pattern_potential(prev_grid)
            next_pattern = pattern_potential(next_grid)
            pattern_change = next_pattern - prev_pattern
            reward += pattern_change * self.pattern_bonus
            
            # Open lines preservation
            prev_open = calculate_open_lines(prev_grid)
            next_open = calculate_open_lines(next_grid)
            open_change = next_open - prev_open
            reward += open_change * self.open_line_bonus
            
            # Evenness improvement (lower is better)
            prev_evenness = calculate_evenness(prev_grid)
            next_evenness = calculate_evenness(next_grid)
            evenness_change = prev_evenness - next_evenness  # Note: inverted
            reward += evenness_change * self.evenness_bonus
            
            # Corner/edge usage improvement
            prev_corner_edge = calculate_corner_edge_usage(prev_grid)
            next_corner_edge = calculate_corner_edge_usage(next_grid)
            corner_edge_change = next_corner_edge - prev_corner_edge
            reward += corner_edge_change * self.corner_edge_bonus
        
        return reward

class CurriculumReward(RewardFunction):
    """
    Curriculum-based reward function that adapts as training progresses.
    Initially provides strong shaping rewards, then gradually transitions to
    sparse rewards as the agent improves.
    """
    
    def __init__(self, initial_reward=None, target_reward=None, curriculum_steps=1000):
        """
        Initialize the curriculum reward function.
        
        Args:
            initial_reward: Initial reward function (defaults to AdvancedReward)
            target_reward: Target reward function (defaults to BasicReward)
            curriculum_steps: Number of updates to transition from initial to target
        """
        super().__init__()
        
        # Set default reward functions if not provided
        self.initial_reward = initial_reward or AdvancedReward()
        self.target_reward = target_reward or BasicReward()
        
        # Curriculum parameters
        self.curriculum_steps = curriculum_steps
        self.current_step = 0
        
        # Initialize interpolation weights
        self.initial_weight = 1.0
        self.target_weight = 0.0
    
    def calculate_reward(self, prev_state, action, next_state, game_info):
        """
        Calculate reward using a weighted average of initial and target rewards.
        
        Args:
            prev_state: Previous game state
            action: Action taken
            next_state: Resulting game state
            game_info: Additional game information
            
        Returns:
            float: Calculated reward
        """
        # Calculate individual rewards
        initial_r = self.initial_reward.calculate_reward(prev_state, action, next_state, game_info)
        target_r = self.target_reward.calculate_reward(prev_state, action, next_state, game_info)
        
        # Return weighted average
        return self.initial_weight * initial_r + self.target_weight * target_r
    
    def update(self):
        """
        Update the curriculum weights based on current step.
        Should be called once per training episode or batch.
        """
        if self.current_step < self.curriculum_steps:
            # Linear interpolation from initial to target
            progress = self.current_step / self.curriculum_steps
            self.initial_weight = 1.0 - progress
            self.target_weight = progress
            
            # Increment step
            self.current_step += 1

class CompositeTieredReward(RewardFunction):
    """
    Composite reward function with multiple tiers that trigger based on state conditions.
    Allows for flexible reward design without complex manual tuning.
    """
    
    def __init__(self):
        """Initialize the composite tiered reward function."""
        super().__init__()
        
        # Base rewards
        self.tier1_reward = {  # Basic gameplay
            'block_placed': 1.0,
            'single_line_clear': 10.0,
            'game_over': -50.0
        }
        
        self.tier2_reward = {  # Strategic gameplay
            'multi_line_clear': 35.0,
            'consecutive_clear': 10.0,
            'isolated_gap': -0.5
        }
        
        self.tier3_reward = {  # Advanced gameplay
            'connectivity_improvement': 0.3,
            'pattern_potential_improvement': 0.3,
            'open_line_preservation': 0.2,
            'evenness_improvement': 0.2,
            'corner_edge_usage': 0.2
        }
        
        # Activation thresholds for tiers
        self.tier2_threshold = 50  # Score needed to activate tier 2
        self.tier3_threshold = 200  # Score needed to activate tier 3
    
    def calculate_reward(self, prev_state, action, next_state, game_info):
        """
        Calculate reward based on active tiers.
        
        Args:
            prev_state: Previous game state
            action: Action taken
            next_state: Resulting game state
            game_info: Dictionary with:
                - score: Current game score
                - blocks_placed: Number of blocks placed
                - rows_cleared: Number of rows cleared
                - cols_cleared: Number of columns cleared
                - consecutive_clears: Number of consecutive clears
                - game_over: Whether the game is over
            
        Returns:
            float: Calculated reward
        """
        reward = 0.0
        score = game_info.get('score', 0)
        
        # Tier 1 rewards (always active)
        blocks_placed = game_info.get('blocks_placed', 0)
        reward += blocks_placed * self.tier1_reward['block_placed']
        
        total_cleared = game_info.get('rows_cleared', 0) + game_info.get('cols_cleared', 0)
        if total_cleared == 1:
            reward += self.tier1_reward['single_line_clear']
        
        if game_info.get('game_over', False):
            reward += self.tier1_reward['game_over']
        
        # Tier 2 rewards (activate after threshold)
        if score >= self.tier2_threshold:
            if total_cleared > 1:
                reward += self.tier2_reward['multi_line_clear']
            
            if game_info.get('consecutive_clears', 0) > 1:
                reward += self.tier2_reward['consecutive_clear']
            
            if 'grid' in next_state:
                isolated_gaps = count_isolated_empty_cells(next_state['grid'])
                reward += isolated_gaps * self.tier2_reward['isolated_gap']
        
        # Tier 3 rewards (activate after threshold)
        if score >= self.tier3_threshold and 'grid' in prev_state and 'grid' in next_state:
            prev_grid = prev_state['grid']
            next_grid = next_state['grid']
            
            # Connectivity improvement
            connectivity_change = calculate_connectivity(next_grid) - calculate_connectivity(prev_grid)
            reward += connectivity_change * self.tier3_reward['connectivity_improvement']
            
            # Pattern potential improvement
            pattern_change = pattern_potential(next_grid) - pattern_potential(prev_grid)
            reward += pattern_change * self.tier3_reward['pattern_potential_improvement']
            
            # Open lines preservation
            open_change = calculate_open_lines(next_grid) - calculate_open_lines(prev_grid)
            reward += open_change * self.tier3_reward['open_line_preservation']
            
            # Evenness improvement (lower is better)
            evenness_change = calculate_evenness(prev_grid) - calculate_evenness(next_grid)
            reward += evenness_change * self.tier3_reward['evenness_improvement']
            
            # Corner/edge usage improvement
            corner_edge_change = calculate_corner_edge_usage(next_grid) - calculate_corner_edge_usage(prev_grid)
            reward += corner_edge_change * self.tier3_reward['corner_edge_usage']
        
        return reward

# Factory function to get reward function by name
def get_reward_function(name="advanced"):
    """
    Factory function to get a reward function by name.
    
    Args:
        name: Name of the reward function
        
    Returns:
        RewardFunction: Initialized reward function
    """
    if name == "basic":
        return BasicReward()
    elif name == "advanced":
        return AdvancedReward()
    elif name == "curriculum":
        return CurriculumReward()
    elif name == "tiered":
        return CompositeTieredReward()
    else:
        raise ValueError(f"Unknown reward function: {name}")