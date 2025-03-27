"""
Shape definitions for Block Puzzle game.
Includes all shapes from the original game implementation.
"""

import numpy as np

# Shape definitions
SHAPES = {
    "single": np.array([[1]]),
    "line_2_horizontal": np.array([[1, 1]]),
    "line_2_vertical": np.array([[1], [1]]),
    "line_3_horizontal": np.array([[1, 1, 1]]),
    "line_3_vertical": np.array([[1], [1], [1]]),
    "line_4_horizontal": np.array([[1, 1, 1, 1]]),
    "line_4_vertical": np.array([[1], [1], [1], [1]]),
    "line_5_horizontal": np.array([[1, 1, 1, 1, 1]]),
    "line_5_vertical": np.array([[1], [1], [1], [1], [1]]),
    "square_2x2": np.array([[1, 1], [1, 1]]),
    "L_shape": np.array([[1, 0], [1, 0], [1, 1]]),
    "L_shape_2": np.array([[0, 1], [0, 1], [1, 1]]),
    "L_shape_flipped": np.array([[1, 1], [1, 0], [1, 0]]),
    "L_shape_2_flipped": np.array([[1, 1], [0, 1], [0, 1]]),
    "L_shape_turn": np.array([[1, 0, 0], [1, 1, 1]]),
    "L_shape_2_turn": np.array([[1, 1, 1], [1, 0, 0]]),
    "L_shape_otherturn": np.array([[1, 1, 1], [0, 0, 1]]),
    "L_shape_2_otherturn": np.array([[0, 0, 1], [1, 1, 1]]),
    "T_shape": np.array([[1, 1, 1], [0, 1, 0]]),
    "T_shape_rotated": np.array([[0, 1], [1, 1], [0, 1]]),
    "T_shape_flipped": np.array([[0, 1, 0], [1, 1, 1]]),
    "T_shape_rotated_flipped": np.array([[1, 0], [1, 1], [1, 0]]),
    "Z_shape": np.array([[1, 1, 0], [0, 1, 1]]),
    "Z_shape_vertical": np.array([[0, 1], [1, 1], [1, 0]]),
    "S_shape": np.array([[0, 1, 1], [1, 1, 0]]),
    "S_shape_vertical": np.array([[1, 0], [1, 1], [0, 1]]),
    "corner_small": np.array([[1, 0], [1, 1]]),
    "corner_small_4": np.array([[0, 1], [1, 1]]),
    "corner_small_2": np.array([[1, 1], [0, 1]]),
    "corner_small_3": np.array([[1, 1], [1, 0]]),
    "big_square": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    "corner_big_1": np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]]),
    "corner_big_2": np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]),
    "corner_big_3": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),
    "corner_big_4": np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]]),
    "big_rectangle": np.array([[1, 1], [1, 1], [1, 1]]),
    "big_rectangle_2": np.array([[1, 1, 1], [1, 1, 1]])
}

# Shape categories for potential curriculum learning
SHAPE_CATEGORIES = {
    "basic": ["single", "line_2_horizontal", "line_2_vertical", "square_2x2"],
    "medium": ["line_3_horizontal", "line_3_vertical", "corner_small", 
               "corner_small_2", "corner_small_3", "corner_small_4"],
    "advanced": ["L_shape", "L_shape_2", "T_shape", "Z_shape", "S_shape"],
    "expert": ["big_square", "line_4_horizontal", "line_4_vertical", 
               "big_rectangle", "big_rectangle_2"],
}

def get_shape_size(shape):
    """Returns the number of filled cells in a shape."""
    return np.sum(shape)

def get_shape_dimensions(shape):
    """Returns the height and width of a shape."""
    return shape.shape

def get_shape_by_name(name):
    """Returns a shape by its name."""
    return SHAPES.get(name, None)

def get_random_shapes(n=3, categories=None):
    """Get n random shapes, optionally filtered by categories."""
    import random
    
    if categories:
        # Flatten the categories to get available shapes
        available_shapes = []
        for category in categories:
            available_shapes.extend(SHAPE_CATEGORIES.get(category, []))
    else:
        available_shapes = list(SHAPES.keys())
    
    # Ensure we have enough shapes
    if len(available_shapes) < n:
        available_shapes = list(SHAPES.keys())
    
    # Select n random shapes
    selected_names = random.sample(available_shapes, n)
    return [SHAPES[name].copy() for name in selected_names], selected_names
