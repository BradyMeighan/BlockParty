"""
Pygame renderer for Block Puzzle game.
This module provides visualization components that can be reused across different scripts.
"""

import pygame
import numpy as np
import random

class PygameRenderer:
    """
    Renderer for Block Puzzle game using Pygame.
    """
    
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
    
    def __init__(self, env, window_size=(800, 600), grid_size=40, grid_margin=1,
                 grid_origin=(30, 30)):
        """
        Initialize the renderer.
        
        Args:
            env: BlockPuzzleEnv environment
            window_size: Tuple of (width, height) for the window
            grid_size: Size of each grid cell in pixels
            grid_margin: Margin between grid cells in pixels
            grid_origin: Tuple of (x, y) for the grid origin
        """
        self.env = env
        self.window_size = window_size
        self.grid_size = grid_size
        self.grid_margin = grid_margin
        self.grid_origin = grid_origin
        
        # Create screen if not already initialized
        if not pygame.get_init():
            pygame.init()
        
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Block Puzzle")
        
        # Generate shape colors
        self.shape_colors = {}
        
        # Initialize fonts
        self.font_small = pygame.font.SysFont('Arial', 18)
        self.font_medium = pygame.font.SysFont('Arial', 24)
        self.font_large = pygame.font.SysFont('Arial', 48)
    
    def draw_board(self):
        """Draw the game board grid."""
        grid_origin_x, grid_origin_y = self.grid_origin
        
        # Draw grid background
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, 
                        (grid_origin_x - 5, grid_origin_y - 5, 
                         self.env.grid_size * (self.grid_size + self.grid_margin) + 10, 
                         self.env.grid_size * (self.grid_size + self.grid_margin) + 10))
        
        # Draw grid cells
        for row in range(self.env.grid_size):
            for col in range(self.env.grid_size):
                x = grid_origin_x + col * (self.grid_size + self.grid_margin)
                y = grid_origin_y + row * (self.grid_size + self.grid_margin)
                
                if self.env.game.grid[row, col] == 1:
                    # Draw filled cell
                    pygame.draw.rect(self.screen, self.DARK_BLUE, (x, y, self.grid_size, self.grid_size))
                else:
                    # Draw empty cell
                    pygame.draw.rect(self.screen, self.WHITE, (x, y, self.grid_size, self.grid_size))
                    # Draw cell border
                    pygame.draw.rect(self.screen, self.GRAY, (x, y, self.grid_size, self.grid_margin), 1)

    def draw_shape_panel(self, selected_shape_idx=None):
        """Draw the panel showing available shapes."""
        grid_origin_x, grid_origin_y = self.grid_origin
        panel_x = grid_origin_x + self.env.grid_size * (self.grid_size + self.grid_margin) + 20
        panel_width = 120
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, 
                        (panel_x - 5, grid_origin_y - 5, 
                         panel_width + 10, 3 * 120))
        
        # Draw title
        title = self.font_small.render("Available Shapes:", True, self.BLACK)
        self.screen.blit(title, (panel_x, grid_origin_y - 30))
        
        # Initialize shape colors once if not already done
        if not hasattr(self.env.game, 'shape_colors'):
            self.env.game.shape_colors = {}
        
        # Draw shapes
        for i, shape in enumerate(self.env.game.current_shapes):
            # Ensure this shape has a stable color
            shape_name = self.env.game.current_shape_names[i]
            if shape_name not in self.env.game.shape_colors:
                self.env.game.shape_colors[shape_name] = random.choice(self.COLORS)
            
            color = self.env.game.shape_colors[shape_name]
            
            # Rest of the drawing code remains the same...
        
        # Draw shapes
        for i, shape in enumerate(self.env.game.current_shapes):
            shape_height, shape_width = shape.shape
            panel_y = grid_origin_y + i * (100 + 20)  # 100px height for each shape + 20px margin
            
            # Draw panel for this shape
            color = self.env.game.shape_colors[self.env.game.current_shape_names[i]]
            
            # Highlight selected shape
            if selected_shape_idx == i:
                pygame.draw.rect(self.screen, self.YELLOW, (panel_x - 2, panel_y - 2, panel_width + 4, 104))
            
            pygame.draw.rect(self.screen, self.WHITE, (panel_x, panel_y, panel_width, 100))
            
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
                        pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))
                        pygame.draw.rect(self.screen, self.BLACK, (x, y, cell_size, cell_size), 1)
    
    def draw_shape_preview(self, shape, row, col):
        """
        Draw a preview of where the shape would be placed.
        
        Args:
            shape: The shape to preview
            row: Row index
            col: Column index
        """
        grid_origin_x, grid_origin_y = self.grid_origin
        shape_height, shape_width = shape.shape
        
        # Check if shape can be placed here
        can_place = self.env.game.can_place_shape(shape, row, col)
        
        # Draw preview cells
        for r in range(shape_height):
            for c in range(shape_width):
                if shape[r, c] == 1:
                    # Calculate position
                    x = grid_origin_x + (col + c) * (self.grid_size + self.grid_margin)
                    y = grid_origin_y + (row + r) * (self.grid_size + self.grid_margin)
                    
                    # Draw with appropriate color
                    if can_place:
                        # Semi-transparent green preview
                        s = pygame.Surface((self.grid_size, self.grid_size), pygame.SRCALPHA)
                        s.fill((0, 255, 0, 128))  # Green with alpha
                        self.screen.blit(s, (x, y))
                    else:
                        # Red indicator that placement is invalid
                        s = pygame.Surface((self.grid_size, self.grid_size), pygame.SRCALPHA)
                        s.fill((255, 0, 0, 128))  # Red with alpha
                        self.screen.blit(s, (x, y))
    
    def draw_score(self):
        """Draw the score display."""
        grid_origin_x, grid_origin_y = self.grid_origin
        
        # Draw score
        score_text = self.font_medium.render(f"Score: {int(self.env.game.score)}", True, self.BLACK)
        self.screen.blit(score_text, (grid_origin_x, grid_origin_y - 30))
        
        # Draw lines cleared
        lines_text = self.font_medium.render(
            f"Lines: {self.env.game.total_rows_cleared + self.env.game.total_cols_cleared}", 
            True, self.BLACK
        )
        self.screen.blit(lines_text, (grid_origin_x + 200, grid_origin_y - 30))
    
    def draw_game_over(self):
        """Draw game over screen."""
        # Create semi-transparent overlay
        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        # Draw game over text
        game_over_text = self.font_large.render("GAME OVER", True, self.RED)
        self.screen.blit(game_over_text, 
                       (self.window_size[0] // 2 - game_over_text.get_width() // 2, 
                        self.window_size[1] // 2 - game_over_text.get_height() // 2))
        
        # Draw final score
        final_score_text = self.font_medium.render(f"Final Score: {int(self.env.game.score)}", True, self.WHITE)
        self.screen.blit(final_score_text, 
                       (self.window_size[0] // 2 - final_score_text.get_width() // 2, 
                        self.window_size[1] // 2 + 50))
        
        # Draw lines cleared
        lines_text = self.font_medium.render(
            f"Lines Cleared: {self.env.game.total_rows_cleared + self.env.game.total_cols_cleared}", 
            True, self.WHITE
        )
        self.screen.blit(lines_text, 
                       (self.window_size[0] // 2 - lines_text.get_width() // 2, 
                        self.window_size[1] // 2 + 80))
        
        # Draw restart instruction
        restart_text = self.font_medium.render("Press any key to play again, ESC to quit", True, self.WHITE)
        self.screen.blit(restart_text, 
                       (self.window_size[0] // 2 - restart_text.get_width() // 2, 
                        self.window_size[1] // 2 + 120))
    
    def draw_instructions(self, text):
        """
        Draw instructions text.
        
        Args:
            text: Instruction text
        """
        grid_origin_x, grid_origin_y = self.grid_origin
        
        instructions = self.font_medium.render(text, True, self.BLACK)
        self.screen.blit(instructions, (grid_origin_x, 
                                      grid_origin_y + self.env.grid_size * (self.grid_size + self.grid_margin) + 10))
    
    def draw_turn_indicator(self, is_human_turn):
        """
        Draw turn indicator for human vs agent mode.
        
        Args:
            is_human_turn: Whether it's the human's turn
        """
        grid_origin_x, grid_origin_y = self.grid_origin
        
        if is_human_turn:
            turn_text = self.font_medium.render("YOUR TURN", True, self.GREEN)
        else:
            turn_text = self.font_medium.render("AGENT TURN", True, self.RED)
        
        self.screen.blit(turn_text, (grid_origin_x + 400, grid_origin_y - 30))
    
    def draw_action_info(self, action):
        """
        Draw information about an action.
        
        Args:
            action: The action to display
        """
        grid_origin_x, grid_origin_y = self.grid_origin
        
        action_text = self.font_medium.render(f"Action: {action}", True, self.BLACK)
        self.screen.blit(action_text, (grid_origin_x, 
                                     grid_origin_y + self.env.grid_size * (self.grid_size + self.grid_margin) + 40))
    
    def clear_screen(self):
        """Clear the screen."""
        self.screen.fill(self.WHITE)
    
    def update_display(self):
        """Update the display."""
        pygame.display.flip()
    
    def handle_events(self):
        """
        Handle pygame events.
        
        Returns:
            tuple: (running, event_data)
            
            running: Whether the game should continue running
            event_data: Dictionary with event data such as mouse_pos, key_pressed, etc.
        """
        event_data = {
            'mouse_pos': None,
            'mouse_click': False,
            'key_pressed': None,
            'quit': False
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                event_data['quit'] = True
                return False, event_data
            
            if event.type == pygame.MOUSEMOTION:
                event_data['mouse_pos'] = event.pos
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                event_data['mouse_click'] = True
                event_data['mouse_pos'] = event.pos
            
            if event.type == pygame.KEYDOWN:
                event_data['key_pressed'] = event.key
        
        return True, event_data
    
    def grid_position_from_mouse(self, mouse_pos):
        """
        Convert mouse position to grid position.
        
        Args:
            mouse_pos: (x, y) mouse position
            
        Returns:
            tuple: (row, col) grid position or None if outside grid
        """
        grid_origin_x, grid_origin_y = self.grid_origin
        mouse_x, mouse_y = mouse_pos
        
        # Check if mouse is over the grid
        if (grid_origin_x <= mouse_x < grid_origin_x + self.env.grid_size * (self.grid_size + self.grid_margin) and
            grid_origin_y <= mouse_y < grid_origin_y + self.env.grid_size * (self.grid_size + self.grid_margin)):
            # Calculate grid position
            grid_x = (mouse_x - grid_origin_x) // (self.grid_size + self.grid_margin)
            grid_y = (mouse_y - grid_origin_y) // (self.grid_size + self.grid_margin)
            return (grid_y, grid_x)  # (row, col)
        
        return None
    
    def shape_selected_from_mouse(self, mouse_pos):
        """
        Determine if a shape was selected based on mouse position.
        
        Args:
            mouse_pos: (x, y) mouse position
            
        Returns:
            int: Index of selected shape or None if no shape was selected
        """
        grid_origin_x, grid_origin_y = self.grid_origin
        mouse_x, mouse_y = mouse_pos
        
        # Check if a shape is selected
        shape_panel_x = grid_origin_x + self.env.grid_size * (self.grid_size + self.grid_margin) + 20
        
        for i in range(len(self.env.game.current_shapes)):
            panel_y = grid_origin_y + i * (100 + 20)
            
            # Check if click is within shape panel
            if (shape_panel_x <= mouse_x < shape_panel_x + 120 and
                panel_y <= mouse_y < panel_y + 100):
                return i
        
        return None
