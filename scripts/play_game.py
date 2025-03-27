#!/usr/bin/env python
"""
Play Block Puzzle game with a trained agent or human input.
This script provides a Pygame interface for playing the game.
"""

import os
import sys
import argparse
import numpy as np
import pygame
import random
import torch

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.block_puzzle_env import BlockPuzzleEnv
from src.agents.base_agent import RandomAgent, HeuristicAgent
from src.agents.ppo_agent import PPOAgent

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 40
GRID_MARGIN = 1
GRID_ORIGIN_X = 30
GRID_ORIGIN_Y = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 139)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
COLORS = [(100, 149, 237), (255, 105, 180), (50, 205, 50), (255, 165, 0), (138, 43, 226)]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Play Block Puzzle game')
    
    # Game parameters
    parser.add_argument('--grid_size', type=int, default=8,
                        help='Size of the square grid (default: 8)')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum steps per episode (default: 200)')
    parser.add_argument('--reward_shaping', action='store_true',
                        help='Use shaped rewards')
    
    # Player mode
    parser.add_argument('--mode', type=str, default='human',
                        choices=['human', 'agent', 'agent_vs_human'],
                        help='Play mode (default: human)')
    
    # Agent parameters
    parser.add_argument('--agent', type=str, default='ppo',
                        choices=['random', 'heuristic', 'ppo'],
                        help='Agent type (default: ppo)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (required for ppo agent)')
    
    # Visualization parameters
    parser.add_argument('--fps', type=int, default=60,
                        help='Frames per second (default: 60)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between agent moves in seconds (default: 0.5)')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['agent', 'agent_vs_human'] and args.agent == 'ppo' and args.model_path is None:
        parser.error("--model_path is required for PPO agent")
    
    return args

def create_agent(args, env):
    """Create an agent based on args."""
    if args.agent == 'random':
        return RandomAgent(env)
    elif args.agent == 'heuristic':
        return HeuristicAgent(env)
    elif args.agent == 'ppo':
        # Create PPO agent
        agent = PPOAgent(env)
        
        # Load model
        agent.load(args.model_path)
        agent.eval()  # Set to evaluation mode
        
        return agent
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

def draw_board(screen, env):
    """Draw the game board grid."""
    # Draw grid background
    pygame.draw.rect(screen, LIGHT_BLUE, 
                    (GRID_ORIGIN_X - 5, GRID_ORIGIN_Y - 5, 
                     env.grid_size * (GRID_SIZE + GRID_MARGIN) + 10, 
                     env.grid_size * (GRID_SIZE + GRID_MARGIN) + 10))
    
    # Draw grid cells
    for row in range(env.grid_size):
        for col in range(env.grid_size):
            x = GRID_ORIGIN_X + col * (GRID_SIZE + GRID_MARGIN)
            y = GRID_ORIGIN_Y + row * (GRID_SIZE + GRID_MARGIN)
            
            if env.game.grid[row, col] == 1:
                # Draw filled cell
                pygame.draw.rect(screen, DARK_BLUE, (x, y, GRID_SIZE, GRID_SIZE))
            else:
                # Draw empty cell
                pygame.draw.rect(screen, WHITE, (x, y, GRID_SIZE, GRID_SIZE))
                # Draw cell border
                pygame.draw.rect(screen, GRAY, (x, y, GRID_SIZE, GRID_SIZE), 1)

def draw_shape_panel(screen, env, selected_shape_idx=None):
    """Draw the panel showing available shapes."""
    panel_x = GRID_ORIGIN_X + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 20
    panel_width = 120
    
    # Draw panel background
    pygame.draw.rect(screen, LIGHT_BLUE, 
                    (panel_x - 5, GRID_ORIGIN_Y - 5, 
                     panel_width + 10, 3 * 120))
    
    # Draw available shapes
    font = pygame.font.SysFont('Arial', 18)
    title = font.render("Available Shapes:", True, BLACK)
    screen.blit(title, (panel_x, GRID_ORIGIN_Y - 30))
    
    # Generate shape colors if needed
    shape_colors = getattr(env.game, 'shape_colors', None)
    if shape_colors is None:
        shape_colors = {}
        for i, name in enumerate(env.game.current_shape_names):
            shape_colors[name] = random.choice(COLORS)
    
    for i, shape in enumerate(env.game.current_shapes):
        shape_height, shape_width = shape.shape
        panel_y = GRID_ORIGIN_Y + i * (100 + 20)  # 100px height for each shape + 20px margin
        
        # Draw panel for this shape
        color = shape_colors.get(env.game.current_shape_names[i], DARK_BLUE)
        
        # Highlight selected shape
        if selected_shape_idx == i:
            pygame.draw.rect(screen, YELLOW, (panel_x - 2, panel_y - 2, panel_width + 4, 104))
        
        pygame.draw.rect(screen, WHITE, (panel_x, panel_y, panel_width, 100))
        
        # Calculate cell size based on shape dimensions
        max_dim = max(shape_height, shape_width)
        cell_size = min(80 // max_dim, 20)  # Ensure it fits in the panel
        
        # Center the shape in the panel
        shape_panel_x = panel_x + (panel_width - shape_width * cell_size) // 2
        shape_panel_y = panel_y + (100 - shape_height * cell_size) // 2
        
        # Draw the shape
        for r in range(shape_height):
            for c in range(shape_width):
                if shape[r, c] == 1:
                    x = shape_panel_x + c * cell_size
                    y = shape_panel_y + r * cell_size
                    pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))
                    pygame.draw.rect(screen, BLACK, (x, y, cell_size, cell_size), 1)

def draw_shape_preview(screen, env, shape, row, col):
    """Draw a preview of where the shape would be placed."""
    shape_height, shape_width = shape.shape
    
    # Check if shape can be placed here
    can_place = env.game.can_place_shape(shape, row, col)
    
    # Draw preview cells
    for r in range(shape_height):
        for c in range(shape_width):
            if shape[r, c] == 1:
                # Calculate position
                x = GRID_ORIGIN_X + (col + c) * (GRID_SIZE + GRID_MARGIN)
                y = GRID_ORIGIN_Y + (row + r) * (GRID_SIZE + GRID_MARGIN)
                
                # Draw with appropriate color
                if can_place:
                    # Semi-transparent green preview
                    s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                    s.fill((0, 255, 0, 128))  # Green with alpha
                    screen.blit(s, (x, y))
                else:
                    # Red indicator that placement is invalid
                    s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                    s.fill((255, 0, 0, 128))  # Red with alpha
                    screen.blit(s, (x, y))

def play_human(env, screen, clock, fps):
    """Play the game with human input."""
    # Game state
    selected_shape_idx = None
    hover_pos = None
    
    # Initialize colors for shapes once at the start
    if not hasattr(env.game, 'shape_colors'):
        env.game.shape_colors = {}
        for name in env.game.current_shape_names:
            env.game.shape_colors[name] = random.choice([
                (100, 149, 237), (255, 105, 180), (50, 205, 50), 
                (255, 165, 0), (138, 43, 226)
            ])
    
    # Game loop
    running = True
    while running and not env.game.game_over:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Mouse movement for shape preview
            if event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                # Check if mouse is over the grid
                if (GRID_ORIGIN_X <= mouse_x < GRID_ORIGIN_X + env.grid_size * (GRID_SIZE + GRID_MARGIN) and
                    GRID_ORIGIN_Y <= mouse_y < GRID_ORIGIN_Y + env.grid_size * (GRID_SIZE + GRID_MARGIN)):
                    # Calculate grid position
                    grid_x = (mouse_x - GRID_ORIGIN_X) // (GRID_SIZE + GRID_MARGIN)
                    grid_y = (mouse_y - GRID_ORIGIN_Y) // (GRID_SIZE + GRID_MARGIN)
                    hover_pos = (grid_y, grid_x)  # (row, col)
                else:
                    hover_pos = None
            
            # Mouse click for shape selection and placement
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                
                # Check if a shape is selected
                shape_panel_x = GRID_ORIGIN_X + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 20
                
                for i in range(len(env.game.current_shapes)):
                    panel_y = GRID_ORIGIN_Y + i * (100 + 20)
                    
                    # Check if click is within shape panel
                    if (shape_panel_x <= mouse_x < shape_panel_x + 120 and
                        panel_y <= mouse_y < panel_y + 100):
                        selected_shape_idx = i
                        break
                
                # If a shape is selected and mouse is over grid, attempt to place it
                if selected_shape_idx is not None and hover_pos is not None:
                    row, col = hover_pos
                    action = (selected_shape_idx, row, col)
                    
                    # Take action in environment
                    obs, reward, done, info = env.step(action)
                    
                    # Reset selection if action was successful
                    if info.get('invalid_action', False) == False:
                        selected_shape_idx = None
                        
            # Add right after a successful move is made
            if env.game.is_game_over():
                env.game.game_over = True
                print("Game over detected!")
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw game board
        draw_board(screen, env)
        
        # Draw shape selection panel
        draw_shape_panel(screen, env, selected_shape_idx)
        
        # Draw score
        font = pygame.font.SysFont('Arial', 24)
        score_text = font.render(f"Score: {int(env.game.score)}", True, BLACK)
        screen.blit(score_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y - 30))
        
        # Draw lines cleared
        lines_text = font.render(
            f"Lines: {env.game.total_rows_cleared + env.game.total_cols_cleared}", 
            True, BLACK
        )
        screen.blit(lines_text, (GRID_ORIGIN_X + 200, GRID_ORIGIN_Y - 30))
        
        # Draw instructions
        instructions = font.render("Select a shape and place it on the grid", True, BLACK)
        screen.blit(instructions, (GRID_ORIGIN_X, GRID_ORIGIN_Y + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 10))
        
        # Draw preview of shape placement
        if selected_shape_idx is not None and hover_pos is not None:
            row, col = hover_pos
            shape = env.game.current_shapes[selected_shape_idx]
            draw_shape_preview(screen, env, shape, row, col)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(fps)
    
    return running

def play_agent(env, agent, screen, clock, fps, delay):
    """Play the game with an agent."""
    # Game loop
    running = True
    while running and not env.game.game_over:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get observation
        obs = env._get_observation()
        
        # Get agent action
        action = agent.act(obs)
        
        # Highlight selected shape
        selected_shape_idx = action[0]
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw game board
        draw_board(screen, env)
        
        # Draw shape selection panel
        draw_shape_panel(screen, env, selected_shape_idx)
        
        # Draw score
        font = pygame.font.SysFont('Arial', 24)
        score_text = font.render(f"Score: {int(env.game.score)}", True, BLACK)
        screen.blit(score_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y - 30))
        
        # Draw lines cleared
        lines_text = font.render(
            f"Lines: {env.game.total_rows_cleared + env.game.total_cols_cleared}", 
            True, BLACK
        )
        screen.blit(lines_text, (GRID_ORIGIN_X + 200, GRID_ORIGIN_Y - 30))
        
        # Draw agent info
        agent_text = font.render(f"Agent: {args.agent}", True, BLACK)
        screen.blit(agent_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 10))
        
        # Draw action info
        action_text = font.render(f"Action: {action}", True, BLACK)
        screen.blit(action_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 40))
        
        # Draw preview of shape placement
        shape = env.game.current_shapes[selected_shape_idx]
        row, col = action[1], action[2]
        draw_shape_preview(screen, env, shape, row, col)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(fps)
        
        # Add delay to make it easier to see agent actions
        pygame.time.delay(int(delay * 1000))
        
        # Take action in environment
        obs, reward, done, info = env.step(action)
    
    return running

def play_agent_vs_human(env, agent, screen, clock, fps, delay):
    """Alternating play between agent and human."""
    # Game state
    human_turn = True
    selected_shape_idx = None
    hover_pos = None
    
    # Game loop
    running = True
    while running and not env.game.game_over:
        if human_turn:
            # Human turn
            font = pygame.font.SysFont('Arial', 24)
            turn_text = font.render("YOUR TURN", True, GREEN)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Mouse movement for shape preview
                if event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.pos
                    # Check if mouse is over the grid
                    if (GRID_ORIGIN_X <= mouse_x < GRID_ORIGIN_X + env.grid_size * (GRID_SIZE + GRID_MARGIN) and
                        GRID_ORIGIN_Y <= mouse_y < GRID_ORIGIN_Y + env.grid_size * (GRID_SIZE + GRID_MARGIN)):
                        # Calculate grid position
                        grid_x = (mouse_x - GRID_ORIGIN_X) // (GRID_SIZE + GRID_MARGIN)
                        grid_y = (mouse_y - GRID_ORIGIN_Y) // (GRID_SIZE + GRID_MARGIN)
                        hover_pos = (grid_y, grid_x)  # (row, col)
                    else:
                        hover_pos = None
                
                # Mouse click for shape selection and placement
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    
                    # Check if a shape is selected
                    shape_panel_x = GRID_ORIGIN_X + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 20
                    
                    for i in range(len(env.game.current_shapes)):
                        panel_y = GRID_ORIGIN_Y + i * (100 + 20)
                        
                        # Check if click is within shape panel
                        if (shape_panel_x <= mouse_x < shape_panel_x + 120 and
                            panel_y <= mouse_y < panel_y + 100):
                            selected_shape_idx = i
                            break
                    
                    # If a shape is selected and mouse is over grid, attempt to place it
                    if selected_shape_idx is not None and hover_pos is not None:
                        row, col = hover_pos
                        action = (selected_shape_idx, row, col)
                        
                        # Take action in environment
                        obs, reward, done, info = env.step(action)
                        
                        # Reset selection if action was successful
                        if info.get('invalid_action', False) == False:
                            selected_shape_idx = None
                            human_turn = False  # Switch to agent turn
        else:
            # Agent turn
            font = pygame.font.SysFont('Arial', 24)
            turn_text = font.render("AGENT TURN", True, RED)
            
            # Handle events (only check for quit)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get observation
            obs = env._get_observation()
            
            # Get agent action
            action = agent.act(obs)
            
            # Highlight selected shape
            selected_shape_idx = action[0]
            row, col = action[1], action[2]
            
            # Draw preview of shape placement
            shape = env.game.current_shapes[selected_shape_idx]
            
            # Update display to show agent's planned move
            screen.fill(WHITE)
            draw_board(screen, env)
            draw_shape_panel(screen, env, selected_shape_idx)
            draw_shape_preview(screen, env, shape, row, col)
            screen.blit(turn_text, (GRID_ORIGIN_X + 400, GRID_ORIGIN_Y - 30))
            
            score_text = font.render(f"Score: {int(env.game.score)}", True, BLACK)
            screen.blit(score_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y - 30))
            
            lines_text = font.render(
                f"Lines: {env.game.total_rows_cleared + env.game.total_cols_cleared}", 
                True, BLACK
            )
            screen.blit(lines_text, (GRID_ORIGIN_X + 200, GRID_ORIGIN_Y - 30))
            
            action_text = font.render(f"Agent action: {action}", True, BLACK)
            screen.blit(action_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 10))
            
            pygame.display.flip()
            clock.tick(fps)
            
            # Add delay to make it easier to see agent actions
            pygame.time.delay(int(delay * 1000))
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            
            # Switch back to human turn
            selected_shape_idx = None
            human_turn = True
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw game board
        draw_board(screen, env)
        
        # Draw shape selection panel
        draw_shape_panel(screen, env, selected_shape_idx)
        
        # Draw score
        font = pygame.font.SysFont('Arial', 24)
        score_text = font.render(f"Score: {int(env.game.score)}", True, BLACK)
        screen.blit(score_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y - 30))
        
        # Draw turn indicator
        screen.blit(turn_text, (GRID_ORIGIN_X + 400, GRID_ORIGIN_Y - 30))
        
        # Draw lines cleared
        lines_text = font.render(
            f"Lines: {env.game.total_rows_cleared + env.game.total_cols_cleared}", 
            True, BLACK
        )
        screen.blit(lines_text, (GRID_ORIGIN_X + 200, GRID_ORIGIN_Y - 30))
        
        # Draw instructions if human turn
        if human_turn:
            instructions = font.render("Select a shape and place it on the grid", True, BLACK)
            screen.blit(instructions, (GRID_ORIGIN_X, GRID_ORIGIN_Y + env.grid_size * (GRID_SIZE + GRID_MARGIN) + 10))
            
            # Draw preview of shape placement
            if selected_shape_idx is not None and hover_pos is not None:
                row, col = hover_pos
                shape = env.game.current_shapes[selected_shape_idx]
                draw_shape_preview(screen, env, shape, row, col)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(fps)
    
    return running

def draw_game_over(screen, env):
    """Draw game over screen."""
    # Create semi-transparent overlay
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    screen.blit(overlay, (0, 0))
    
    # Draw game over text
    font_big = pygame.font.SysFont('Arial', 48)
    game_over_text = font_big.render("GAME OVER", True, RED)
    screen.blit(game_over_text, 
               (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2))
    
    # Draw final score
    font = pygame.font.SysFont('Arial', 24)
    final_score_text = font.render(f"Final Score: {int(env.game.score)}", True, WHITE)
    screen.blit(final_score_text, 
               (SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, 
                SCREEN_HEIGHT // 2 + 50))
    
    # Draw lines cleared
    lines_text = font.render(
        f"Lines Cleared: {env.game.total_rows_cleared + env.game.total_cols_cleared}", 
        True, WHITE
    )
    screen.blit(lines_text, 
               (SCREEN_WIDTH // 2 - lines_text.get_width() // 2, 
                SCREEN_HEIGHT // 2 + 80))
    
    # Draw restart instruction
    restart_text = font.render("Press any key to play again, ESC to quit", True, WHITE)
    screen.blit(restart_text, 
               (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                SCREEN_HEIGHT // 2 + 120))

def main():
    """Main function."""
    # Parse arguments
    global args
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Block Puzzle")
    clock = pygame.time.Clock()
    
    # Main game loop
    running = True
    while running:
        # Create environment
        env = BlockPuzzleEnv(
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            reward_shaping=args.reward_shaping
        )
        
        # Reset environment
        obs = env.reset()
        
        # Create agent if needed
        agent = None
        if args.mode in ['agent', 'agent_vs_human']:
            agent = create_agent(args, env)
        
        # Play the game
        if args.mode == 'human':
            running = play_human(env, screen, clock, args.fps)
        elif args.mode == 'agent':
            running = play_agent(env, agent, screen, clock, args.fps, args.delay)
        elif args.mode == 'agent_vs_human':
            running = play_agent_vs_human(env, agent, screen, clock, args.fps, args.delay)
        
        # Game over - wait for restart
        if running and env.game.game_over:
            draw_game_over(screen, env)
            pygame.display.flip()
            
            waiting_for_key = True
            while waiting_for_key and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting_for_key = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        waiting_for_key = False
                
                clock.tick(args.fps)
    
    # Quit pygame
    pygame.quit()

if __name__ == '__main__':
    main()
