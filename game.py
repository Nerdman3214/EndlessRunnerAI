import pygame
import random
import sys
from collections import namedtuple
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_BLUE = (135, 206, 235)
GROUND_COLOR = (210, 180, 140)
PLAYER_COLOR = (255, 50, 50)
OBSTACLE_COLOR = (34, 139, 34)

# Game variables
GRAVITY = 0.8
JUMP_STRENGTH = -15
GROUND_HEIGHT = 300
OBSTACLE_SPEED = 5
OBSTACLE_SPAWN_RATE = 90  # Lower = more frequent
GAME_OVER_DELAY = 180  # Frames to wait before auto-restart (3 seconds at 60 FPS)

Point = namedtuple('Point', 'x y')


class AI(pygame.sprite.Sprite):
    """Player character controlled by AI or keyboard input."""
    
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 60))
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = 100
        self.rect.bottom = GROUND_HEIGHT
        self.velocity_y = 0
        self.is_jumping = False

    def jump(self):
        """Make the player jump if on the ground."""
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True

    def update(self):
        """Update player physics (gravity and position)."""
        # Apply gravity
        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y

        # Check if player is on the ground
        if self.rect.bottom >= GROUND_HEIGHT:
            self.rect.bottom = GROUND_HEIGHT
            self.velocity_y = 0
            self.is_jumping = False


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x):
        super().__init__()
        self.width = random.randint(20, 40)
        self.height = random.randint(40, 80)
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(OBSTACLE_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.bottom = GROUND_HEIGHT

    def update(self):
        self.rect.x -= OBSTACLE_SPEED
        # Remove obstacle if it goes off screen
        if self.rect.right < 0:
            self.kill()


class Cloud(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((60, 30))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(SCREEN_WIDTH, SCREEN_WIDTH + 200)
        self.rect.y = random.randint(30, 150)
        self.speed = random.uniform(0.5, 2)

    def update(self):
        self.rect.x -= self.speed
        if self.rect.right < 0:
            self.rect.x = SCREEN_WIDTH
            self.rect.y = random.randint(30, 150)


class Game:
    """Main game class with AI training interface."""
    
    def __init__(self, use_display=True):
        """Initialize the game.
        
        Args:
            use_display: If False, runs in headless mode (faster for training)
        """
        self.use_display = use_display
        self.running = True  # Initialize running flag
        
        if use_display:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH,
                                                   SCREEN_HEIGHT))
            pygame.display.set_caption("Endless Runner")
        else:
            # Headless mode for faster training
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.game_over_font = pygame.font.Font(None, 72)
        self.reset_game()

    def reset_game(self):
        """Reset the game to initial state. Called at start and after game over."""
        self.player = AI()
        self.all_sprites = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.clouds = pygame.sprite.Group()
        
        self.all_sprites.add(self.player)
        
        # Add some clouds for visual appeal
        for _ in range(3):
            cloud = Cloud()
            self.clouds.add(cloud)
            self.all_sprites.add(cloud)
        
        self.score = 0
        self.obstacle_timer = 0
        self.running = True
        self.game_over = False
        self.game_over_timer = 0
        self.frame_iteration = 0  # Track total frames for this game

    def handle_events(self):
        """Handle pygame events (quit, keyboard input for manual play)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.game_over:
                        self.player.jump()
                    else:
                        self.reset_game()
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True

    def spawn_obstacle(self):
        """Spawn obstacles at regular intervals."""
        self.obstacle_timer += 1
        if self.obstacle_timer >= OBSTACLE_SPAWN_RATE:
            obstacle = Obstacle(SCREEN_WIDTH)
            self.obstacles.add(obstacle)
            self.all_sprites.add(obstacle)
            self.obstacle_timer = 0

    def is_collision(self):
        """Check if player has collided with an obstacle.
        
        Returns:
            bool: True if collision occurred, False otherwise
        """
        # Check if player collides with any obstacle
        hits = pygame.sprite.spritecollide(self.player, self.obstacles, False)
        return len(hits) > 0

    def get_nearest_obstacle(self):
        """Find the nearest obstacle ahead of the player.
        
        Returns:
            Obstacle or None: The nearest obstacle, or None if no obstacles exist
        """
        nearest = None
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            # Only consider obstacles ahead of the player
            if obstacle.rect.x > self.player.rect.x:
                distance = obstacle.rect.x - self.player.rect.x
                if distance < min_distance:
                    min_distance = distance
                    nearest = obstacle
        
        return nearest
    
    def get_state(self):
        """Get the current game state for AI input.
        
        Returns:
            numpy array: State vector with 6 features:
                [0] Player Y position (normalized)
                [1] Player Y velocity (normalized)
                [2] Is player jumping (1 or 0)
                [3] Distance to nearest obstacle (normalized)
                [4] Nearest obstacle height (normalized)
                [5] Nearest obstacle width (normalized)
        """
        nearest_obstacle = self.get_nearest_obstacle()
        
        # Normalize player position (0 to 1)
        player_y_norm = self.player.rect.y / GROUND_HEIGHT
        
        # Normalize velocity (-1 to 1 approximately)
        velocity_norm = self.player.velocity_y / 20.0
        
        # Is jumping (0 or 1)
        is_jumping = 1 if self.player.is_jumping else 0
        
        if nearest_obstacle:
            # Distance to obstacle (normalized by screen width)
            distance_x = nearest_obstacle.rect.x - self.player.rect.x
            distance = distance_x / SCREEN_WIDTH
            # Normalize obstacle dimensions
            obstacle_height = nearest_obstacle.height / 100.0
            obstacle_width = nearest_obstacle.width / 50.0
        else:
            # No obstacle ahead - use safe default values
            distance = 1.0
            obstacle_height = 0.0
            obstacle_width = 0.0
        
        state = np.array([
            player_y_norm,
            velocity_norm,
            is_jumping,
            distance,
            obstacle_height,
            obstacle_width
        ], dtype=float)
        
        return state
    
    def play_step(self, action=None):
        """Execute one game step. This is the main method for AI training.
        
        Args:
            action: 0 = do nothing, 1 = jump (if None, uses keyboard input)
        
        Returns:
            tuple: (reward, game_over, score)
                - reward (float): Reward for this step
                - game_over (bool): Whether the game ended
                - score (int): Current score
        """
        self.frame_iteration += 1
        reward = 0
        
        # Handle events (quit, manual control)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # Execute action
        if action is not None:
            # AI mode: action is 0 (do nothing) or 1 (jump)
            if action == 1:
                self.player.jump()
        else:
            # Manual mode: check keyboard
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.player.jump()
        
        # Update game state
        self.all_sprites.update()
        self.spawn_obstacle()
        
        # Small reward for each frame survived
        reward = 0.1
        
        # Check for collision
        if self.is_collision():
            self.game_over = True
            reward = -10  # Large penalty for dying
            return reward, True, self.score
        
        # Increase score (each frame survived)
        self.score += 1
        
        # Small bonus for successfully passing an obstacle
        nearest = self.get_nearest_obstacle()
        if nearest and nearest.rect.right < self.player.rect.left:
            # Just passed an obstacle
            reward = 1.0
        
        return reward, False, self.score
    
    def update(self):
        """Update game state for manual play mode (with auto-reset)."""
        if not self.game_over:
            self.all_sprites.update()
            self.spawn_obstacle()
            if self.is_collision():
                self.game_over = True
            self.score += 1
        else:
            # Auto-reset after delay
            self.game_over_timer += 1
            if self.game_over_timer >= GAME_OVER_DELAY:
                self.reset_game()

    def draw(self):
        """Render the game to the screen."""
        if not self.use_display:
            return  # Skip rendering in headless mode
        
        # Draw sky
        self.screen.fill(SKY_BLUE)
        
        # Draw ground
        ground_rect = (0, GROUND_HEIGHT, SCREEN_WIDTH,
                       SCREEN_HEIGHT - GROUND_HEIGHT)
        pygame.draw.rect(self.screen, GROUND_COLOR, ground_rect)
        
        # Draw all sprites
        self.all_sprites.draw(self.screen)
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        
        # Draw instructions
        if self.score < 100:
            instruction = "Press SPACE to jump"
            instruction_text = self.font.render(instruction, True, BLACK)
            self.screen.blit(instruction_text, (SCREEN_WIDTH - 300, 10))
        
        # Draw game over screen
        if self.game_over:
            game_over_text = self.game_over_font.render("GAME OVER",
                                                         True, BLACK)
            score_msg = f"Final Score: {self.score}"
            final_score_text = self.font.render(score_msg, True, BLACK)
            
            # Calculate countdown
            seconds_left = (GAME_OVER_DELAY - self.game_over_timer) // FPS + 1
            restart_msg = f"Restarting in {seconds_left}..."
            restart_text = self.font.render(restart_msg, True, BLACK)
            
            # Center the text
            x_pos = SCREEN_WIDTH // 2 - game_over_text.get_width() // 2
            self.screen.blit(game_over_text, (x_pos, SCREEN_HEIGHT // 2 - 60))
            
            x_pos = SCREEN_WIDTH // 2 - final_score_text.get_width() // 2
            self.screen.blit(final_score_text, (x_pos, SCREEN_HEIGHT // 2))
            
            x_pos = SCREEN_WIDTH // 2 - restart_text.get_width() // 2
            self.screen.blit(restart_text, (x_pos, SCREEN_HEIGHT // 2 + 40))
        
        pygame.display.flip()

    def run(self):
        """Main game loop for manual play (not AI training).
        
        This runs the traditional game loop with keyboard controls.
        For AI training, use play_step() instead.
        """
        while self.running:
            self.running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # Manual play mode - run the game with keyboard controls
    game = Game(use_display=True)
    game.run()
