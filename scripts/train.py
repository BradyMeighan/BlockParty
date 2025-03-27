#!/usr/bin/env python
"""
Training script for Block Puzzle RL agents.
"""

import os
import sys
import argparse
import numpy as np
import time
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.block_puzzle_env import BlockPuzzleEnv
from src.agents.base_agent import RandomAgent, HeuristicAgent
from src.agents.ppo_agent import PPOAgent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train an RL agent for Block Puzzle')
    
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
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer dimension for neural networks (default: 128)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    
    # PPO-specific parameters
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO clipping parameter (default: 0.2)')
    parser.add_argument('--target_kl', type=float, default=0.01,
                        help='Target KL divergence (default: 0.01)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter (default: 0.95)')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient (default: 0.01)')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes (default: 10000)')
    parser.add_argument('--update_interval', type=int, default=10,
                        help='Update interval in episodes (default: 10)')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval in episodes (default: 100)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Model save interval in episodes (default: 1000)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='data/trained_models',
                        help='Directory to save models and logs (default: data/trained_models)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: agent_type_timestamp)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model to resume training from')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Set experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.agent}_{timestamp}"
    
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
        return PPOAgent(
            env=env,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            clip_ratio=args.clip_ratio,
            target_kl=args.target_kl,
            gae_lambda=args.gae_lambda,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate an agent's performance."""
    agent.eval()  # Set to evaluation mode
    
    scores = []
    rows_cleared = []
    cols_cleared = []
    episode_lengths = []
    
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            score += reward
            steps += 1
        
        scores.append(info['score'])
        rows_cleared.append(info['rows_cleared'])
        cols_cleared.append(info['cols_cleared'])
        episode_lengths.append(steps)
    
    agent.train()  # Set back to training mode
    
    return {
        'mean_score': np.mean(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_rows_cleared': np.mean(rows_cleared),
        'mean_cols_cleared': np.mean(cols_cleared),
        'mean_episode_length': np.mean(episode_lengths)
    }

def main():
    """Main training function."""
    # Parse arguments
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
    
    # Resume training if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        agent.load(args.resume)
    
    # Initialize metrics
    metrics = {
        'episodes': [],
        'train_scores': [],
        'train_rows_cleared': [],
        'train_cols_cleared': [],
        'train_episode_lengths': [],
        'eval_metrics': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'kls': [],
        'clip_fractions': []
    }
    
    # Training loop
    print(f"Starting training with agent: {args.agent}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    for episode in range(args.episodes):
        # Initialize episode
        obs = env.reset()
        done = False
        episode_score = 0
        episode_steps = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.act(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Update stats
            episode_score += reward
            episode_steps += 1
            
            # Update observation
            obs = next_obs
        
        # Update metrics
        metrics['episodes'].append(episode)
        metrics['train_scores'].append(info['score'])
        metrics['train_rows_cleared'].append(info['rows_cleared'])
        metrics['train_cols_cleared'].append(info['cols_cleared'])
        metrics['train_episode_lengths'].append(episode_steps)
        
        # Update agent (for PPO)
        if args.agent == 'ppo' and (episode + 1) % args.update_interval == 0:
            update_metrics = agent.update()
            
            metrics['policy_losses'].append(update_metrics.get('policy_loss', 0))
            metrics['value_losses'].append(update_metrics.get('value_loss', 0))
            metrics['entropies'].append(update_metrics.get('entropy', 0))
            metrics['kls'].append(update_metrics.get('kl', 0))
            metrics['clip_fractions'].append(update_metrics.get('clip_fraction', 0))
        
        # Evaluate agent
        if (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_agent(agent, env, args.eval_episodes)
            metrics['eval_metrics'].append({
                'episode': episode,
                **eval_metrics
            })
            
            if args.verbose:
                print(f"Episode {episode + 1}/{args.episodes} | " +
                      f"Score: {info['score']:.2f} | " +
                      f"Rows cleared: {info['rows_cleared']} | " +
                      f"Cols cleared: {info['cols_cleared']} | " +
                      f"Eval score: {eval_metrics['mean_score']:.2f}")
        elif args.verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{args.episodes} | " +
                  f"Score: {info['score']:.2f} | " +
                  f"Rows cleared: {info['rows_cleared']} | " +
                  f"Cols cleared: {info['cols_cleared']}")
        
        # Save model
        if (episode + 1) % args.save_interval == 0:
            model_path = os.path.join(output_dir, f"model_{episode + 1}.pt")
            agent.save(model_path)
            
            # Save metrics
            metrics_path = os.path.join(output_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    # Save final model
    model_path = os.path.join(output_dir, "model_final.pt")
    agent.save(model_path)
    
    # Save final metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save args
    args_path = os.path.join(output_dir, "args.json")
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Print final stats
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final evaluation score: {metrics['eval_metrics'][-1]['mean_score']:.2f}")
    print(f"Models and metrics saved to {output_dir}")

if __name__ == '__main__':
    main()
