#!/usr/bin/env python
"""
Evaluation script for Block Puzzle RL agents.
This script evaluates trained agents and generates performance metrics.
"""

import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.block_puzzle_env import BlockPuzzleEnv
from src.agents.base_agent import RandomAgent, HeuristicAgent
from src.agents.ppo_agent import PPOAgent
from src.utils.state_utils import analyze_state

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate an RL agent for Block Puzzle')
    
    # Environment parameters
    parser.add_argument('--grid_size', type=int, default=8,
                        help='Size of the square grid (default: 8)')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum steps per episode (default: 200)')
    parser.add_argument('--reward_shaping', action='store_true',
                        help='Use shaped rewards')
    
    # Agent parameters
    parser.add_argument('--agent', type=str, default='ppo',
                        choices=['random', 'heuristic', 'ppo'],
                        help='Agent type (default: ppo)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (required for ppo agent)')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize evaluation episodes')
    parser.add_argument('--visualize_count', type=int, default=5,
                        help='Number of episodes to visualize (default: 5)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for visualization (default: 30)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='data/evaluations',
                        help='Directory to save evaluation results (default: data/evaluations)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: agent_type_timestamp)')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Validation
    if args.agent == 'ppo' and args.model_path is None:
        parser.error("--model_path is required for ppo agent")
    
    # Set experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"eval_{args.agent}_{timestamp}"
    
    return args

def set_seed(seed):
    """Set random seeds."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_agent(args, env):
    """Create an agent based on args."""
    if args.agent == 'random':
        return RandomAgent(env)
    elif args.agent == 'heuristic':
        return HeuristicAgent(env)
    elif args.agent == 'ppo':
        agent = PPOAgent(env)
        agent.load(args.model_path)
        agent.eval()  # Set to evaluation mode
        return agent
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

def evaluate_episode(env, agent, render=False, fps=30):
    """
    Evaluate a single episode.
    
    Args:
        env: The environment
        agent: The agent to evaluate
        render: Whether to render the episode
        fps: Frames per second for rendering
        
    Returns:
        dict: Episode results
    """
    # Initialize pygame for rendering if needed
    if render:
        import pygame
        from src.visualization.pygame_renderer import PygameRenderer
        
        if not pygame.get_init():
            pygame.init()
        
        renderer = PygameRenderer(env)
        clock = pygame.time.Clock()
    
    # Reset environment
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    
    # Track metrics
    actions = []
    rewards = []
    grid_states = []
    valid_moves_counts = []
    
    # Episode loop
    running = True
    while running and not done:
        # Handle events if rendering
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Render state
            renderer.clear_screen()
            renderer.draw_board()
            renderer.draw_shape_panel()
            renderer.draw_score()
            renderer.draw_instructions(f"Agent: {args.agent}")
            renderer.update_display()
            clock.tick(fps)
        
        # Get valid moves count
        valid_moves = env.get_valid_actions()
        valid_moves_counts.append(len(valid_moves))
        
        # Select action
        action = agent.act(obs)
        actions.append(action)
        
        # Render action preview if rendering
        if render:
            shape_idx, row, col = action
            shape = env.game.current_shapes[shape_idx]
            
            renderer.clear_screen()
            renderer.draw_board()
            renderer.draw_shape_panel(selected_shape_idx=shape_idx)
            renderer.draw_score()
            renderer.draw_shape_preview(shape, row, col)
            renderer.draw_action_info(action)
            renderer.update_display()
            
            # Add slight delay to see the action
            pygame.time.delay(200)
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        # Update metrics
        episode_reward += reward
        episode_steps += 1
        rewards.append(reward)
        
        # Save grid state for analysis
        grid_states.append(env.game.grid.copy())
        
        # Update observation
        obs = next_obs
    
    # Final render if rendering
    if render:
        renderer.clear_screen()
        renderer.draw_board()
        renderer.draw_shape_panel()
        renderer.draw_score()
        
        if env.game.game_over:
            renderer.draw_game_over()
        
        renderer.update_display()
        pygame.time.delay(1000)  # Pause for 1 second to see the final state
    
    # Analyze the episode
    episode_analysis = analyze_episode(
        grid_states=grid_states,
        actions=actions,
        rewards=rewards,
        valid_moves_counts=valid_moves_counts
    )
    
    # Return results
    return {
        'score': env.game.score,
        'reward': episode_reward,
        'steps': episode_steps,
        'rows_cleared': env.game.total_rows_cleared,
        'cols_cleared': env.game.total_cols_cleared,
        'game_over': env.game.game_over,
        'analysis': episode_analysis
    }

def analyze_episode(grid_states, actions, rewards, valid_moves_counts):
    """
    Analyze an episode's trajectory.
    
    Args:
        grid_states: List of grid states
        actions: List of actions taken
        rewards: List of rewards received
        valid_moves_counts: List of valid moves counts
        
    Returns:
        dict: Analysis results
    """
    # Calculate grid fill rate over time
    fill_rates = [np.mean(grid) for grid in grid_states]
    
    # Calculate reward distribution
    reward_distribution = {
        'min': min(rewards) if rewards else 0,
        'max': max(rewards) if rewards else 0,
        'mean': np.mean(rewards) if rewards else 0,
        'median': np.median(rewards) if rewards else 0,
        'std': np.std(rewards) if rewards else 0
    }
    
    # Analyze valid moves over time
    valid_moves_stats = {
        'min': min(valid_moves_counts) if valid_moves_counts else 0,
        'max': max(valid_moves_counts) if valid_moves_counts else 0,
        'mean': np.mean(valid_moves_counts) if valid_moves_counts else 0,
        'median': np.median(valid_moves_counts) if valid_moves_counts else 0,
        'std': np.std(valid_moves_counts) if valid_moves_counts else 0
    }
    
    # Analyze action distributions
    action_distributions = {}
    if actions:
        # Shape usage
        shape_usage = {}
        for action in actions:
            shape_idx = action[0]
            shape_usage[shape_idx] = shape_usage.get(shape_idx, 0) + 1
        
        # Position distribution
        row_distribution = {}
        col_distribution = {}
        for action in actions:
            _, row, col = action
            row_distribution[row] = row_distribution.get(row, 0) + 1
            col_distribution[col] = col_distribution.get(col, 0) + 1
        
        action_distributions = {
            'shape_usage': shape_usage,
            'row_distribution': row_distribution,
            'col_distribution': col_distribution
        }
    
    return {
        'fill_rates': fill_rates,
        'reward_distribution': reward_distribution,
        'valid_moves_stats': valid_moves_stats,
        'action_distributions': action_distributions
    }

def aggregate_results(results):
    """
    Aggregate evaluation results.
    
    Args:
        results: List of episode results
        
    Returns:
        dict: Aggregated results
    """
    # Basic metrics
    scores = [r['score'] for r in results]
    rewards = [r['reward'] for r in results]
    steps = [r['steps'] for r in results]
    rows_cleared = [r['rows_cleared'] for r in results]
    cols_cleared = [r['cols_cleared'] for r in results]
    total_cleared = [r['rows_cleared'] + r['cols_cleared'] for r in results]
    game_overs = [r['game_over'] for r in results]
    
    # Aggregate analysis
    fill_rates = []
    for r in results:
        fill_rates.extend(r['analysis']['fill_rates'])
    
    valid_moves_mins = [r['analysis']['valid_moves_stats']['min'] for r in results]
    valid_moves_means = [r['analysis']['valid_moves_stats']['mean'] for r in results]
    
    # Combine action distributions
    shape_usage = {}
    row_distribution = {}
    col_distribution = {}
    
    for r in results:
        for shape_idx, count in r['analysis']['action_distributions'].get('shape_usage', {}).items():
            shape_usage[shape_idx] = shape_usage.get(shape_idx, 0) + count
        
        for row, count in r['analysis']['action_distributions'].get('row_distribution', {}).items():
            row_distribution[row] = row_distribution.get(row, 0) + count
        
        for col, count in r['analysis']['action_distributions'].get('col_distribution', {}).items():
            col_distribution[col] = col_distribution.get(col, 0) + count
    
    # Return aggregated results
    return {
        'score': {
            'min': min(scores),
            'max': max(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores)
        },
        'reward': {
            'min': min(rewards),
            'max': max(rewards),
            'mean': np.mean(rewards),
            'median': np.median(rewards),
            'std': np.std(rewards)
        },
        'steps': {
            'min': min(steps),
            'max': max(steps),
            'mean': np.mean(steps),
            'median': np.median(steps),
            'std': np.std(steps)
        },
        'rows_cleared': {
            'min': min(rows_cleared),
            'max': max(rows_cleared),
            'mean': np.mean(rows_cleared),
            'median': np.median(rows_cleared),
            'std': np.std(rows_cleared)
        },
        'cols_cleared': {
            'min': min(cols_cleared),
            'max': max(cols_cleared),
            'mean': np.mean(cols_cleared),
            'median': np.median(cols_cleared),
            'std': np.std(cols_cleared)
        },
        'total_cleared': {
            'min': min(total_cleared),
            'max': max(total_cleared),
            'mean': np.mean(total_cleared),
            'median': np.median(total_cleared),
            'std': np.std(total_cleared)
        },
        'completion_rate': 1 - sum(game_overs) / len(game_overs),
        'fill_rates': {
            'min': min(fill_rates) if fill_rates else 0,
            'max': max(fill_rates) if fill_rates else 0,
            'mean': np.mean(fill_rates) if fill_rates else 0,
            'std': np.std(fill_rates) if fill_rates else 0
        },
        'valid_moves': {
            'min_of_mins': min(valid_moves_mins) if valid_moves_mins else 0,
            'mean_of_means': np.mean(valid_moves_means) if valid_moves_means else 0
        },
        'action_distributions': {
            'shape_usage': shape_usage,
            'row_distribution': row_distribution,
            'col_distribution': col_distribution
        }
    }

def generate_plots(results, aggregated_results, output_dir):
    """
    Generate evaluation plots.
    
    Args:
        results: List of episode results
        aggregated_results: Aggregated results
        output_dir: Output directory
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Score distribution plot
    plt.figure(figsize=(10, 6))
    scores = [r['score'] for r in results]
    plt.hist(scores, bins=20, alpha=0.7)
    plt.axvline(aggregated_results['score']['mean'], color='r', linestyle='--', 
               label=f"Mean: {aggregated_results['score']['mean']:.2f}")
    plt.axvline(aggregated_results['score']['median'], color='g', linestyle='--', 
               label=f"Median: {aggregated_results['score']['median']:.2f}")
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'score_distribution.png'))
    plt.close()
    
    # Lines cleared distribution
    plt.figure(figsize=(10, 6))
    total_cleared = [r['rows_cleared'] + r['cols_cleared'] for r in results]
    plt.hist(total_cleared, bins=20, alpha=0.7)
    plt.axvline(aggregated_results['total_cleared']['mean'], color='r', linestyle='--', 
               label=f"Mean: {aggregated_results['total_cleared']['mean']:.2f}")
    plt.xlabel('Lines Cleared')
    plt.ylabel('Frequency')
    plt.title('Lines Cleared Distribution')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'lines_cleared_distribution.png'))
    plt.close()
    
    # Episode length distribution
    plt.figure(figsize=(10, 6))
    steps = [r['steps'] for r in results]
    plt.hist(steps, bins=20, alpha=0.7)
    plt.axvline(aggregated_results['steps']['mean'], color='r', linestyle='--', 
               label=f"Mean: {aggregated_results['steps']['mean']:.2f}")
    plt.xlabel('Episode Length (steps)')
    plt.ylabel('Frequency')
    plt.title('Episode Length Distribution')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'episode_length_distribution.png'))
    plt.close()
    
    # Action distributions
    plt.figure(figsize=(10, 6))
    shape_usage = aggregated_results['action_distributions']['shape_usage']
    shapes = list(shape_usage.keys())
    counts = [shape_usage[s] for s in shapes]
    plt.bar(shapes, counts)
    plt.xlabel('Shape Index')
    plt.ylabel('Usage Count')
    plt.title('Shape Usage Distribution')
    plt.savefig(os.path.join(plots_dir, 'shape_usage_distribution.png'))
    plt.close()
    
    # Grid position distributions (heatmap)
    plt.figure(figsize=(8, 8))
    row_dist = aggregated_results['action_distributions']['row_distribution']
    col_dist = aggregated_results['action_distributions']['col_distribution']
    
    # Create grid position heatmap
    grid_size = max(max(row_dist.keys(), default=0), max(col_dist.keys(), default=0)) + 1
    position_matrix = np.zeros((grid_size, grid_size))
    
    for r in results:
        for action in r['analysis']['action_distributions'].get('row_distribution', {}):
            _, row, col = action
            if 0 <= row < grid_size and 0 <= col < grid_size:
                position_matrix[row, col] += 1
    
    plt.imshow(position_matrix, cmap='viridis')
    plt.colorbar(label='Placement Frequency')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Grid Position Usage Heatmap')
    plt.savefig(os.path.join(plots_dir, 'position_heatmap.png'))
    plt.close()

def main():
    """Main evaluation function."""
    # Parse arguments
    global args
    args = parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create environment
    env = BlockPuzzleEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        reward_shaping=args.reward_shaping
    )
    
    # Create agent
    agent = create_agent(args, env)
    
    # Evaluate agent
    print(f"Evaluating {args.agent} agent for {args.episodes} episodes...")
    
    # Determine number of episodes to visualize
    visualize_count = min(args.visualize_count, args.episodes) if args.visualize else 0
    
    # Run evaluation
    results = []
    visualize_episodes = set(np.random.choice(range(args.episodes), 
                                            visualize_count, replace=False)) if visualize_count > 0 else set()
    
    for episode in tqdm(range(args.episodes)):
        render = episode in visualize_episodes
        result = evaluate_episode(env, agent, render=render, fps=args.fps)
        results.append(result)
        
        if args.verbose:
            print(f"Episode {episode + 1}/{args.episodes}: " +
                  f"Score: {result['score']:.2f}, " +
                  f"Lines: {result['rows_cleared'] + result['cols_cleared']}, " +
                  f"Steps: {result['steps']}")
    
    # Aggregate results
    aggregated_results = aggregate_results(results)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Mean Score: {aggregated_results['score']['mean']:.2f} ± {aggregated_results['score']['std']:.2f}")
    print(f"Mean Lines Cleared: {aggregated_results['total_cleared']['mean']:.2f} ± {aggregated_results['total_cleared']['std']:.2f}")
    print(f"Mean Episode Length: {aggregated_results['steps']['mean']:.2f} steps")
    print(f"Completion Rate: {aggregated_results['completion_rate'] * 100:.2f}%")
    
    # Generate plots
    generate_plots(results, aggregated_results, output_dir)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays/values to Python native types for JSON serialization
        serializable_results = []
        for r in results:
            # Convert numpy arrays to lists
            r['analysis']['fill_rates'] = r['analysis']['fill_rates'].tolist() \
                if hasattr(r['analysis']['fill_rates'], 'tolist') else r['analysis']['fill_rates']
            
            # Convert numpy values to native Python types
            for key in r['analysis']['reward_distribution']:
                r['analysis']['reward_distribution'][key] = float(r['analysis']['reward_distribution'][key])
            
            for key in r['analysis']['valid_moves_stats']:
                r['analysis']['valid_moves_stats'][key] = float(r['analysis']['valid_moves_stats'][key])
            
            serializable_results.append(r)
        
        # Serialize aggregated results
        serializable_aggregated = {}
        for key, value in aggregated_results.items():
            if isinstance(value, dict):
                serializable_aggregated[key] = {}
                for k, v in value.items():
                    serializable_aggregated[key][k] = float(v) if isinstance(v, np.number) else v
            else:
                serializable_aggregated[key] = float(value) if isinstance(value, np.number) else value
        
        json.dump({
            'results': serializable_results,
            'aggregated': serializable_aggregated,
            'config': vars(args)
        }, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()