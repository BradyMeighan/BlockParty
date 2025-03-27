"""
Environment module for Block Puzzle.
Contains the game logic and RL environment implementation.
"""

from .block_puzzle_env import BlockPuzzleEnv
from .block_puzzle_game import BlockPuzzleGame
from .shapes import SHAPES, get_random_shapes, get_shape_size
