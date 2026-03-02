"""
Deep Q-Learning Agent for Endless Runner

This agent learns to play the endless runner game using:
- Deep Q-Learning algorithm
- Experience replay memory
- Epsilon-greedy exploration
- GPU acceleration (if available)

The agent observes the game state and learns which actions
(jump or don't jump) lead to higher rewards.
"""

import torch
import random
import numpy as np
from collections import deque
from game import Game
from model import Linear_QNet, QTrainer
from helper import plot

# Hyperparameters - Optimized for 100+ scores
MAX_MEMORY = 100_000  # Maximum experiences to store
BATCH_SIZE = 2000        # Larger batch = more stable learning
LR = 0.001               # Learning rate (keep standard)


class Agent:
    """
    AI Agent that learns to play endless runner using Deep Q-Learning.
    
    State Space (6 features):
        - Player Y position
        - Player Y velocity
        - Is jumping (0 or 1)
        - Distance to nearest obstacle
        - Obstacle height
        - Obstacle width
    
    Action Space (2 actions):
        - 0: Don't jump
        - 1: Jump
    """
    
    def __init__(self):
        self.n_games = 0                        # Number of games played
        self.epsilon = 0                        # Exploration rate (randomness)
        self.gamma = 0.95                       # Higher discount = longer-term planning
        self.epsilon_min = 0.01                 # Minimum exploration rate
        self.epsilon_decay = 0.995              # Gradual decay rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay buffer
        
        # Neural network: 6 inputs → 512 hidden → 2 outputs (larger network)
        self.model = Linear_QNet(6, 512, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        print("Agent initialized!")
        print("State size: 6, Action size: 2")
        print("Hidden layer: 512 neurons")
        print(f"Memory capacity: {MAX_MEMORY:,} experiences")
        print(f"Batch size: {BATCH_SIZE:,} | Gamma: {self.gamma}")

    def get_state(self, game):
        """
        This method is NOT used - kept for compatibility.
        Instead, we use game.get_state() directly.
        
        Args:
            game: Game instance
        
        Returns:
            Game state from game.get_state()
        """
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in replay memory.
        
        Experience = (state, action, reward, next_state, done)
        
        Args:
            state: State before action
            action: Action taken (0 or 1)
            reward: Reward received
            next_state: State after action
            done: Whether game ended (True/False)
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Train on a batch of experiences from memory (experience replay).
        
        This helps:
        - Break correlation between consecutive experiences
        - Learn from past experiences multiple times
        - Stabilize training
        """
        if len(self.memory) > BATCH_SIZE:
            # Sample random batch from memory
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            # Use all available experiences if less than batch size
            mini_sample = self.memory

        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Train on batch
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train on a single recent experience (online learning).
        
        This provides immediate feedback for the latest action.
        
        Args:
            state: Current state
            action: Action taken (0 or 1)
            reward: Reward received
            next_state: Next state
            done: Whether game ended
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Choose action using epsilon-greedy strategy.
        
        Epsilon-greedy balances:
        - Exploration: Random actions (try new things)
        - Exploitation: Use model prediction (use what we learned)
        
        As games increase, epsilon decreases → more exploitation
        
        Args:
            state: Current game state (numpy array)
        
        Returns:
            int: Action (0 = don't jump, 1 = jump)
        """
        # Exploration rate decreases gradually (exponential decay)
        # Starts at 100% exploration, decays to 1% minimum
        # This allows continuous learning even atigh game counts
        epsilon_value = max(self.epsilon_min, 1.0 * (self.epsilon_decay ** self.n_games))
        
        # Random action (exploration)
        if random.random() < epsilon_value:
            move = random.randint(0, 1)  # Random: 0 or 1
        else:
            # Use model prediction (exploitation)
            # Move state to same device as model
            device = next(self.model.parameters()).device
            state_tensor = torch.tensor(state, dtype=torch.float).to(device)
            
            # Get Q-values from model
            prediction = self.model(state_tensor)
            
            # Choose action with highest Q-value
            move = torch.argmax(prediction).item()
        
        return move

    
def train():
    """
    Main training loop for the AI agent.
    
    Process:
    1. Agent observes state
    2. Agent chooses action (jump or not)
    3. Game executes action and returns reward
    4. Agent trains on this experience
    5. Repeat until game over
    6. Train on batch of past experiences
    7. Start new game
    
    The agent improves by learning which actions lead to higher rewards.
    """
    # Track scores for plotting
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    
    # Initialize game and agent
    game = Game(use_display=True)  # Set to False for faster training
    agent = Agent()
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    print("Press Ctrl+C to stop training\n")
    
    try:
        while True:
            # Reset game for new episode
            game.reset_game()
            state_old = game.get_state()
            
            episode_reward = 0
            
            # Game loop
            while True:
                # Get action from agent
                action = agent.get_action(state_old)
                
                # Perform action in game
                reward, done, score = game.play_step(action)
                state_new = game.get_state()
                
                episode_reward += reward
                
                # Train short memory (immediate learning)
                agent.train_short_memory(state_old, action, reward, 
                                        state_new, done)
                
                # Remember this experience
                agent.remember(state_old, action, reward, state_new, done)
                
                # Update state
                state_old = state_new
                
                # Render game (optional - comment out for faster training)
                game.draw()
                game.clock.tick(60)  # 60 FPS
                
                # Check if game is over
                if done:
                    break
            
            # Game finished - train on batch of experiences
            agent.n_games += 1
            agent.train_long_memory()
            
            # Track and save best score
            if score > best_score:
                best_score = score
                agent.model.save('best_model.pth')
            
            # Update statistics
            total_score += score
            mean_score = total_score / agent.n_games
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            
            # Display progress
            epsilon_display = max(agent.epsilon_min, 1.0 * (agent.epsilon_decay ** agent.n_games))
            print(f'Game: {agent.n_games:4d} | '
                  f'Score: {score:5d} | '
                  f'Best: {best_score:5d} | '
                  f'Mean: {mean_score:6.1f} | '
                  f'Epsilon: {epsilon_display:.3f} | '
                  f'Memory: {len(agent.memory):6d}')
            
            # Plot progress (updates in real-time)
            plot(plot_scores, plot_mean_scores)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("TRAINING INTERRUPTED")
        print("="*60)
        print(f"Games played: {agent.n_games}")
        print(f"Best score: {best_score}")
        print(f"Final mean score: {mean_score:.2f}")
        print(f"Model saved to: ./model/best_model.pth")
        print("="*60)


if __name__ == '__main__':
    train()