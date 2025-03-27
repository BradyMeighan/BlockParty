import pygame
import numpy as np
import random
import sys

# Initialize pygame
pygame.init()

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

# Shape definitions (from your code)
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

class BlockPuzzleGame:
    def __init__(self):
        self.grid = np.zeros((8, 8), dtype=int)
        self.score = 0
        self.consecutive_clears = 0
        self.current_shapes = []
        self.current_shape_names = []
        self.game_over = False
        self.total_rows_cleared = 0
        self.total_cols_cleared = 0
        self.selected_shape_idx = None
        self.hover_pos = None
        self.cells_to_highlight = []
        self.highlight_timer = 0

        # Reward shaping parameters (same as in original code)
        self.block_reward = 1.0
        self.base_line_clear_rewards = {1: 10, 2: 25}
        self.multi_line_bonus = 50
        self.additional_line_bonus = 10
        self.streak_bonus_factor = 5
        self.gap_penalty_coef = 0.5
        self.game_over_penalty = 50.0

        # Generate shape colors
        self.shape_colors = {}
        for shape_name in SHAPES.keys():
            self.shape_colors[shape_name] = random.choice(COLORS)

        self.draw_new_shapes()

    def draw_new_shapes(self):
        """Draw 3 random shapes from the shape list if there are none available."""
        if not self.current_shapes:
            shape_names = list(SHAPES.keys())
            self.current_shape_names = random.sample(shape_names, 3)
            self.current_shapes = [SHAPES[name].copy() for name in self.current_shape_names]

    def can_place_shape(self, shape, row, col):
        """Check if a shape can be placed at the given position on the grid."""
        shape_height, shape_width = shape.shape

        if row < 0 or col < 0 or row + shape_height > 8 or col + shape_width > 8:
            return False

        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1 and self.grid[row + i, col + j] == 1:
                    return False

        return True

    def place_shape(self, shape_idx, row, col):
        """Place the selected shape on the grid."""
        if shape_idx >= len(self.current_shapes) or self.game_over:
            return False

        shape = self.current_shapes[shape_idx]
        if not self.can_place_shape(shape, row, col):
            return False

        shape_height, shape_width = shape.shape
        blocks_placed = 0
        filled_cells = []
        
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1:
                    self.grid[row + i, col + j] = 1
                    blocks_placed += 1
                    filled_cells.append((row + i, col + j))

        # Highlight placed cells
        self.cells_to_highlight = filled_cells
        self.highlight_timer = 10

        # Compute reward components
        reward = blocks_placed * self.block_reward

        # Check for completed rows and columns
        rows_to_clear, cols_to_clear = self.get_lines_to_clear()
        rows_cleared, cols_cleared = len(rows_to_clear), len(cols_to_clear)
        
        # Add to highlight before clearing
        for row in rows_to_clear:
            for col in range(8):
                if (row, col) not in self.cells_to_highlight:
                    self.cells_to_highlight.append((row, col))
        
        for col in cols_to_clear:
            for row in range(8):
                if (row, col) not in self.cells_to_highlight:
                    self.cells_to_highlight.append((row, col))
        
        # Clear the lines
        self.clear_lines(rows_to_clear, cols_to_clear)
        
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

        # Penalize for empty gaps
        gaps = self.count_empty_gaps()
        reward -= gaps * self.gap_penalty_coef

        # Update score
        self.score += reward

        # Remove the used shape
        self.current_shapes.pop(shape_idx)
        self.current_shape_names.pop(shape_idx)
        self.selected_shape_idx = None

        # If no shapes remain, draw new ones and check for game over
        if not self.current_shapes:
            self.draw_new_shapes()
            if self.is_game_over():
                self.game_over = True
                reward -= self.game_over_penalty
                self.score -= self.game_over_penalty

        return True

    def get_lines_to_clear(self):
        """Identify rows and columns that should be cleared."""
        rows_to_clear = []
        cols_to_clear = []

        # Check rows
        for i in range(8):
            if np.all(self.grid[i, :] == 1):
                rows_to_clear.append(i)

        # Check columns
        for j in range(8):
            if np.all(self.grid[:, j] == 1):
                cols_to_clear.append(j)

        return rows_to_clear, cols_to_clear

    def clear_lines(self, rows_to_clear, cols_to_clear):
        """Clear the specified rows and columns."""
        for row in rows_to_clear:
            self.grid[row, :] = 0

        for col in cols_to_clear:
            self.grid[:, col] = 0

    def is_game_over(self):
        """Check if no valid moves exist."""
        # Two-step check like in the original code
        # First, a quick check with sampling
        for shape_idx, shape in enumerate(self.current_shapes):
            shape_height, shape_width = shape.shape
            step = 2 if shape_height < 3 and shape_width < 3 else 1
            for row in range(0, 8 - shape_height + 1, step):
                for col in range(0, 8 - shape_width + 1, step):
                    if self.can_place_shape(shape, row, col):
                        return False
        
        # Then a thorough check if needed
        for shape_idx, shape in enumerate(self.current_shapes):
            shape_height, shape_width = shape.shape
            for row in range(0, 8 - shape_height + 1):
                for col in range(0, 8 - shape_width + 1):
                    if self.can_place_shape(shape, row, col):
                        return False
        
        return True

    def count_empty_gaps(self):
        """Count isolated empty cells (hard to fill gaps)."""
        gaps = 0
        for i in range(8):
            for j in range(8):
                if self.grid[i, j] == 0:
                    surrounded = 0
                    if i == 0 or self.grid[i - 1, j] == 1:
                        surrounded += 1
                    if i == 7 or self.grid[i + 1, j] == 1:
                        surrounded += 1
                    if j == 0 or self.grid[i, j - 1] == 1:
                        surrounded += 1
                    if j == 7 or self.grid[i, j + 1] == 1:
                        surrounded += 1
                    if surrounded >= 3:
                        gaps += 1
        return gaps

def draw_board(screen, game):
    """Draw the game board grid."""
    # Draw grid background
    pygame.draw.rect(screen, LIGHT_BLUE, 
                    (GRID_ORIGIN_X - 5, GRID_ORIGIN_Y - 5, 
                     8 * (GRID_SIZE + GRID_MARGIN) + 10, 8 * (GRID_SIZE + GRID_MARGIN) + 10))
    
    # Draw grid cells
    for row in range(8):
        for col in range(8):
            x = GRID_ORIGIN_X + col * (GRID_SIZE + GRID_MARGIN)
            y = GRID_ORIGIN_Y + row * (GRID_SIZE + GRID_MARGIN)
            
            # Check if this cell should be highlighted
            is_highlighted = False
            if game.highlight_timer > 0:
                if (row, col) in game.cells_to_highlight:
                    is_highlighted = True
                    pygame.draw.rect(screen, YELLOW, (x, y, GRID_SIZE, GRID_SIZE))
            
            if not is_highlighted:
                if game.grid[row, col] == 1:
                    # Draw filled cell
                    pygame.draw.rect(screen, DARK_BLUE, (x, y, GRID_SIZE, GRID_SIZE))
                else:
                    # Draw empty cell
                    pygame.draw.rect(screen, WHITE, (x, y, GRID_SIZE, GRID_SIZE))
                    # Draw cell border
                    pygame.draw.rect(screen, GRAY, (x, y, GRID_SIZE, GRID_SIZE), 1)

def draw_shape_panel(screen, game):
    """Draw the panel showing available shapes."""
    panel_x = GRID_ORIGIN_X + 8 * (GRID_SIZE + GRID_MARGIN) + 20
    panel_width = 120
    
    # Draw panel background
    pygame.draw.rect(screen, LIGHT_BLUE, 
                    (panel_x - 5, GRID_ORIGIN_Y - 5, 
                     panel_width + 10, 3 * 120))
    
    # Draw available shapes
    font = pygame.font.SysFont('Arial', 18)
    title = font.render("Available Shapes:", True, BLACK)
    screen.blit(title, (panel_x, GRID_ORIGIN_Y - 30))
    
    for i, shape in enumerate(game.current_shapes):
        shape_height, shape_width = shape.shape
        panel_y = GRID_ORIGIN_Y + i * (100 + 20)  # 100px height for each shape + 20px margin
        
        # Draw panel for this shape
        color = game.shape_colors[game.current_shape_names[i]]
        # Highlight selected shape
        if game.selected_shape_idx == i:
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

def draw_shape_preview(screen, shape, row, col, game):
    """Draw a preview of where the shape would be placed."""
    shape_height, shape_width = shape.shape
    
    # Check if shape can be placed here
    can_place = game.can_place_shape(shape, row, col)
    
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

def main():
    # Create screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Block Puzzle")
    
    # Create game instance
    game = BlockPuzzleGame()
    
    # Create clock for controlling frame rate
    clock = pygame.time.Clock()
    
    # Animation states
    show_game_over = False
    game_over_alpha = 0
    
    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not game.game_over:
                # Mouse movement for shape preview
                if event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.pos
                    # Check if mouse is over the grid
                    if (GRID_ORIGIN_X <= mouse_x < GRID_ORIGIN_X + 8 * (GRID_SIZE + GRID_MARGIN) and
                        GRID_ORIGIN_Y <= mouse_y < GRID_ORIGIN_Y + 8 * (GRID_SIZE + GRID_MARGIN)):
                        # Calculate grid position
                        grid_x = (mouse_x - GRID_ORIGIN_X) // (GRID_SIZE + GRID_MARGIN)
                        grid_y = (mouse_y - GRID_ORIGIN_Y) // (GRID_SIZE + GRID_MARGIN)
                        game.hover_pos = (grid_y, grid_x)  # (row, col)
                    else:
                        game.hover_pos = None
                
                # Mouse click for shape selection and placement
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    
                    # Check if a shape is selected
                    shape_panel_x = GRID_ORIGIN_X + 8 * (GRID_SIZE + GRID_MARGIN) + 20
                    
                    for i in range(len(game.current_shapes)):
                        shape = game.current_shapes[i]
                        panel_y = GRID_ORIGIN_Y + i * (100 + 20)
                        
                        # Check if click is within shape panel
                        if (shape_panel_x <= mouse_x < shape_panel_x + 120 and
                            panel_y <= mouse_y < panel_y + 100):
                            game.selected_shape_idx = i
                            break
                    
                    # If a shape is selected and mouse is over grid, attempt to place it
                    if game.selected_shape_idx is not None and game.hover_pos is not None:
                        row, col = game.hover_pos
                        game.place_shape(game.selected_shape_idx, row, col)
            else:
                # If game is over, handle restart
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Reset game
                    game = BlockPuzzleGame()
                    show_game_over = False
                    game_over_alpha = 0
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw game board
        draw_board(screen, game)
        
        # Draw shape selection panel
        draw_shape_panel(screen, game)
        
        # Draw score
        font = pygame.font.SysFont('Arial', 24)
        score_text = font.render(f"Score: {int(game.score)}", True, BLACK)
        screen.blit(score_text, (GRID_ORIGIN_X, GRID_ORIGIN_Y - 30))
        
        # Draw lines cleared
        lines_text = font.render(f"Lines: {game.total_rows_cleared + game.total_cols_cleared}", True, BLACK)
        screen.blit(lines_text, (GRID_ORIGIN_X + 200, GRID_ORIGIN_Y - 30))
        
        # Draw instructions
        instructions = font.render("Select a shape and place it on the grid", True, BLACK)
        screen.blit(instructions, (GRID_ORIGIN_X, GRID_ORIGIN_Y + 8 * (GRID_SIZE + GRID_MARGIN) + 10))
        
        # Draw preview of shape placement
        if game.selected_shape_idx is not None and game.hover_pos is not None:
            row, col = game.hover_pos
            shape = game.current_shapes[game.selected_shape_idx]
            draw_shape_preview(screen, shape, row, col, game)
        
        # Update highlight timer
        if game.highlight_timer > 0:
            game.highlight_timer -= 1
        
        # Handle game over state
        if game.game_over and not show_game_over:
            show_game_over = True
        
        if show_game_over:
            # Create semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            if game_over_alpha < 128:
                game_over_alpha += 4  # Fade in effect
            overlay.fill((0, 0, 0, game_over_alpha))
            screen.blit(overlay, (0, 0))
            
            # Draw game over text
            font_big = pygame.font.SysFont('Arial', 48)
            game_over_text = font_big.render("GAME OVER", True, RED)
            screen.blit(game_over_text, 
                       (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                        SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2))
            
            # Draw final score
            final_score_text = font.render(f"Final Score: {int(game.score)}", True, WHITE)
            screen.blit(final_score_text, 
                       (SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, 
                        SCREEN_HEIGHT // 2 + 50))
            
            # Draw restart instruction
            restart_text = font.render("Click anywhere to play again", True, WHITE)
            screen.blit(restart_text, 
                       (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                        SCREEN_HEIGHT // 2 + 100))
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(60)
    
    # Quit pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()