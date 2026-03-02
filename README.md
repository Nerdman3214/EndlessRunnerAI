# Endless Runner Game

A simple endless runner game built with Pygame where you control a character that must jump over obstacles to survive as long as possible.

**NEW: Now includes AI training interface!** Train your own AI agent using Deep Q-Learning to master the game.

## Features

### Game Features
- **Simple Controls**: Press SPACE to jump
- **Dynamic Obstacles**: Randomly generated obstacles with varying sizes
- **Scoring System**: Your score increases the longer you survive
- **Animated Background**: Moving clouds for a more immersive experience
- **Game Over & Restart**: Easy restart with SPACE after game over
- **Auto-Loop Mode**: Endless automatic restart for AI training

### AI Training Features
- **State-based Interface**: Get game state as 6-feature vector
- **Reward System**: Balanced rewards for survival, obstacle avoidance, and penalties for collision
- **Action Interface**: Simple 0/1 action space (don't jump / jump)
- **Headless Mode**: Run without display for faster training
- **Experience Replay**: Store and sample game experiences
- **Example Templates**: Complete examples for AI training

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Play (Manual Mode)

1. Run the game:

```bash
python game.py
```

2. **Controls**:
   - Press `SPACE` to make your character jump
   - Press `ESC` to quit the game
   - After game over, the game auto-restarts in 3 seconds

3. **Objective**: 
   - Jump over the green obstacles
   - Survive as long as possible to get the highest score
   - Don't let your character collide with any obstacles!

## Game Mechanics

- **Player**: Red rectangle that can jump
- **Obstacles**: Green rectangles of varying sizes that scroll from right to left
- **Clouds**: White decorative elements in the background
- **Ground**: Brown platform at the bottom of the screen
- **Score**: Increases continuously as you play

## AI Training Interface

### Quick Start for AI Training

1. **See examples** of the AI interface:
```bash
python ai_interface_example.py
```

2. **Use the template** to build your AI:
```bash
python agent_template.py
```

### Key Methods for AI Training

#### `game.reset_game()`
Reset the game to initial state. Call this to start a new episode.

#### `game.get_state()` → numpy array[6]
Get current game state as a 6-element vector:
- `[0]` Player Y position (normalized 0-1)
- `[1]` Player Y velocity (normalized)
- `[2]` Is jumping (0 or 1)
- `[3]` Distance to nearest obstacle (normalized 0-1)
- `[4]` Nearest obstacle height (normalized)
- `[5]` Nearest obstacle width (normalized)

#### `game.play_step(action)` → (reward, done, score)
Execute one game step with the given action.

**Args:**
- `action`: 0 (don't jump) or 1 (jump)

**Returns:**
- `reward` (float): Reward for this step
  - +0.1 for surviving each frame
  - +1.0 for passing an obstacle
  - -10.0 for collision (game over)
- `done` (bool): Whether the game ended
- `score` (int): Current score (frames survived)

### Example AI Training Loop

```python
from game import Game
import numpy as np

# Initialize game (headless mode for speed)
game = Game(use_display=False)

for episode in range(1000):
    game.reset_game()
    state = game.get_state()
    
    while True:
        # Your AI decides action based on state
        action = your_model.predict(state)  # 0 or 1
        
        # Execute action
        reward, done, score = game.play_step(action)
        next_state = game.get_state()
        
        # Train your model
        your_model.train(state, action, reward, next_state, done)
        
        state = next_state
        if done:
            break
```

### Files

- **game.py** - Main game with AI interface
- **ai_interface_example.py** - Examples showing how to use the AI interface
- **agent_template.py** - Template for building a Deep Q-Learning agent
- **README.md** - This file

### Training Tips

1. **Start with headless mode**: Use `Game(use_display=False)` for faster training
2. **Monitor mean score**: Track average score over last 100 games to see improvement
3. **Adjust hyperparameters**: Learning rate, epsilon decay, hidden layer size
4. **Save your best model**: Save when you see score improvements
5. **Test with display**: Watch your trained agent play with `use_display=True`

## Customization

You can modify the game by adjusting constants in the code:

- `OBSTACLE_SPEED`: Change how fast obstacles move (default: 5)
- `OBSTACLE_SPAWN_RATE`: Change how frequently obstacles appear (default: 90)
- `JUMP_STRENGTH`: Adjust jump height (default: -15)
- `GRAVITY`: Modify gravity strength (default: 0.8)
- Colors and window size can also be customized

## Requirements

- Python 3.7+
- Pygame 2.5.0+

## Tips

- Timing is everything! Wait for the right moment to jump
- Some obstacles are taller than others - plan your jumps accordingly
- The longer you survive, the higher your score!

Enjoy the game! 🎮
