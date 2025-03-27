"""
Utility functions for state processing and feature extraction.
"""

import numpy as np
from scipy.ndimage import label

def count_filled_cells(grid):
    """
    Count the number of filled cells in the grid.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        int: Number of filled cells
    """
    return np.sum(grid)

def count_empty_cells(grid):
    """
    Count the number of empty cells in the grid.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        int: Number of empty cells
    """
    return grid.size - np.sum(grid)

def fill_percentage(grid):
    """
    Calculate the percentage of filled cells in the grid.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        float: Percentage of filled cells (0-1)
    """
    return np.sum(grid) / grid.size

def row_completeness(grid):
    """
    Calculate the completeness of each row.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        numpy.ndarray: Array of row completeness values (0-1)
    """
    return np.sum(grid, axis=1) / grid.shape[1]

def col_completeness(grid):
    """
    Calculate the completeness of each column.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        numpy.ndarray: Array of column completeness values (0-1)
    """
    return np.sum(grid, axis=0) / grid.shape[0]

def pattern_potential(grid):
    """
    Calculate a metric for how close rows and columns are to being completed.
    Higher values indicate rows/columns that are closer to completion.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        float: Pattern potential score
    """
    # Calculate completeness
    row_comp = row_completeness(grid)
    col_comp = col_completeness(grid)
    
    # Calculate potential score with bias towards almost complete lines
    # (use a non-linear function to emphasize higher completeness values)
    row_potential = np.sum(row_comp ** 2)
    col_potential = np.sum(col_comp ** 2)
    
    return row_potential + col_potential

def count_isolated_empty_cells(grid):
    """
    Count the number of isolated empty cells (empty cells that are surrounded
    by filled cells or borders on 3 or more sides).
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        int: Number of isolated empty cells
    """
    grid_size = grid.shape[0]
    isolated_count = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 0:
                surrounded = 0
                
                # Check top
                if i == 0 or grid[i - 1, j] == 1:
                    surrounded += 1
                
                # Check bottom
                if i == grid_size - 1 or grid[i + 1, j] == 1:
                    surrounded += 1
                
                # Check left
                if j == 0 or grid[i, j - 1] == 1:
                    surrounded += 1
                
                # Check right
                if j == grid_size - 1 or grid[i, j + 1] == 1:
                    surrounded += 1
                
                # If surrounded on 3 or more sides, count as isolated
                if surrounded >= 3:
                    isolated_count += 1
    
    return isolated_count

def calculate_open_lines(grid):
    """
    Calculate the number of rows and columns with 2 or fewer filled cells.
    These are considered "open lines" that are easy to place shapes in.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        int: Number of open lines
    """
    # Count rows with 2 or fewer filled cells
    open_rows = np.sum(np.sum(grid, axis=1) <= 2)
    
    # Count columns with 2 or fewer filled cells
    open_cols = np.sum(np.sum(grid, axis=0) <= 2)
    
    return open_rows + open_cols

def calculate_evenness(grid):
    """
    Calculate the evenness of the grid, measured as the standard deviation of filled
    cells across rows and columns. Lower values indicate more even distribution.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        float: Evenness metric
    """
    # Calculate row and column sums
    row_sums = np.sum(grid, axis=1)
    col_sums = np.sum(grid, axis=0)
    
    # Calculate standard deviations
    row_std = np.std(row_sums)
    col_std = np.std(col_sums)
    
    # Return average standard deviation
    return (row_std + col_std) / 2

def calculate_connectivity(grid):
    """
    Calculate a connectivity score for the filled cells in the grid.
    Higher scores indicate better connected filled cells.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        float: Connectivity score
    """
    # Use scipy's label function to identify connected components
    labeled_array, num_features = label(grid)
    
    # If no filled cells, return 0
    if num_features == 0:
        return 0
    
    # Count the size of each connected component
    component_sizes = []
    for i in range(1, num_features + 1):
        component_sizes.append(np.sum(labeled_array == i))
    
    # Calculate connectivity metrics
    total_filled = np.sum(grid)
    largest_component = max(component_sizes)
    largest_ratio = largest_component / total_filled if total_filled > 0 else 0
    
    # Connectivity score is a combination of the number of components and the size of the largest component
    return largest_ratio * (1 - 0.1 * (num_features - 1))

def calculate_corner_edge_usage(grid):
    """
    Calculate a metric for how well corners and edges are utilized.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        float: Corner and edge usage score (0-1)
    """
    grid_size = grid.shape[0]
    
    # Define corners and edges
    corners = [
        (0, 0), (0, grid_size - 1), 
        (grid_size - 1, 0), (grid_size - 1, grid_size - 1)
    ]
    
    edges = []
    for i in range(1, grid_size - 1):
        edges.extend([(0, i), (i, 0), (grid_size - 1, i), (i, grid_size - 1)])
    
    # Count filled corners and edges
    corner_fill = sum(grid[i, j] for i, j in corners)
    edge_fill = sum(grid[i, j] for i, j in edges)
    
    # Calculate usage scores
    corner_usage = corner_fill / len(corners)
    edge_usage = edge_fill / len(edges)
    
    # Combined score with higher weight for corners
    return 0.6 * corner_usage + 0.4 * edge_usage

def calculate_clearance_potential(grid):
    """
    Calculate the potential for clearing lines, based on how many rows and columns
    are close to being filled.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        float: Clearance potential score
    """
    grid_size = grid.shape[0]
    
    # Calculate row and column sums
    row_sums = np.sum(grid, axis=1)
    col_sums = np.sum(grid, axis=0)
    
    # Calculate potential scores
    row_potential = 0
    col_potential = 0
    
    for count in row_sums:
        if count >= grid_size - 2:  # 1-2 cells away from clearing
            row_potential += (count / grid_size) ** 2
    
    for count in col_sums:
        if count >= grid_size - 2:  # 1-2 cells away from clearing
            col_potential += (count / grid_size) ** 2
    
    return row_potential + col_potential

def extract_state_features(grid):
    """
    Extract a comprehensive set of features from the grid state.
    
    Args:
        grid: 2D numpy array representing the grid
        
    Returns:
        dict: Dictionary of extracted features
    """
    grid_size = grid.shape[0]
    
    # Basic metrics
    filled_cells = count_filled_cells(grid)
    empty_cells = count_empty_cells(grid)
    fill_pct = fill_percentage(grid)
    
    # Line metrics
    row_comp = row_completeness(grid)
    col_comp = col_completeness(grid)
    pattern_pot = pattern_potential(grid)
    open_lines = calculate_open_lines(grid)
    
    # Spatial metrics
    isolated_cells = count_isolated_empty_cells(grid)
    evenness = calculate_evenness(grid)
    connectivity = calculate_connectivity(grid)
    corner_edge_usage = calculate_corner_edge_usage(grid)
    clearance_potential = calculate_clearance_potential(grid)
    
    # Compile into feature dictionary
    features = {
        'filled_cells': filled_cells,
        'empty_cells': empty_cells,
        'fill_percentage': fill_pct,
        'row_completeness': row_comp,
        'col_completeness': col_comp,
        'pattern_potential': pattern_pot,
        'open_lines': open_lines,
        'isolated_cells': isolated_cells,
        'evenness': evenness,
        'connectivity': connectivity,
        'corner_edge_usage': corner_edge_usage,
        'clearance_potential': clearance_potential,
    }
    
    return features

def flattened_state_array(grid, shapes):
    """
    Create a flattened numpy array representation of the state.
    
    Args:
        grid: 2D numpy array representing the grid
        shapes: List of shape arrays
        
    Returns:
        numpy.ndarray: Flattened state array
    """
    # Flatten grid
    grid_flat = grid.flatten()
    
    # Flatten and pad shapes
    shapes_flat = []
    for shape in shapes:
        shapes_flat.extend(shape.flatten())
    
    # Combine and return
    return np.concatenate([grid_flat, np.array(shapes_flat)])

def analyze_state(grid, agent_view=False):
    """
    Analyze a state for evaluation or visualization purposes.
    
    Args:
        grid: 2D numpy array representing the grid
        agent_view: Whether to include additional features useful for agent training
        
    Returns:
        dict: Analysis results
    """
    # Extract basic features
    features = extract_state_features(grid)
    
    # Add derived metrics
    features['playability'] = features['open_lines'] * 0.5 + features['connectivity'] * 0.3 + \
                            (1 - features['evenness'] / 10) * 0.2
    
    features['clearing_opportunity'] = features['clearance_potential'] * 0.6 + \
                                      features['pattern_potential'] * 0.4
    
    features['penalty_factor'] = features['isolated_cells'] * 0.8 + \
                              (1 - features['corner_edge_usage']) * 0.2
    
    # For agent training, include additional features
    if agent_view:
        # Add grid patterns or other complex features
        # This would be expanded based on what helps the agent learn
        pass
    
    return features