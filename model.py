"""
Neural Network Model for Deep Q-Learning

This module contains:
- Linear_QNet: The neural network that predicts Q-values
- QTrainer: Handles training with GPU support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """
    Deep Q-Network with 2 hidden layers.
    
    Architecture:
        Input Layer → Hidden Layer (ReLU) → Hidden Layer (ReLU) → Output Layer
    
    Args:
        input_size: Number of state features (6 for endless runner)
        hidden_size: Number of neurons in hidden layer
        output_size: Number of possible actions (2: don't jump, jump)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
        
        Returns:
            Q-values for each action
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)  # No activation on output (Q-values can be negative)
        return x

    def save(self, file_name='model.pth'):
        """
        Save model weights to disk.
        
        Args:
            file_name: Name of file to save (default: 'model.pth')
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print(f"Model saved to {file_name}")


class QTrainer:
    """
    Trainer for Deep Q-Learning with GPU support.
    
    Handles:
    - Moving data to GPU if available
    - Computing loss using Bellman equation
    - Updating network weights
    
    Args:
        model: The Q-Network to train
        lr: Learning rate
        gamma: Discount factor for future rewards
    """
    
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # GPU Support - automatically use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Print device info
        if torch.cuda.is_available():
            print(f"🚀 GPU Training Enabled: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("💻 Training on CPU (GPU not available)")

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one training step using the Bellman equation.
        
        Q_new = reward + gamma * max(Q(next_state)) if not done
        Q_new = reward if done
        
        Args:
            state: Current state(s)
            action: Action(s) taken
            reward: Reward(s) received
            next_state: Next state(s)
            done: Whether episode(s) ended
        """
        # Convert to tensors and move to GPU
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # Handle single sample (add batch dimension)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Predict Q-values for current state
        pred = self.model(state)

        # 2. Calculate target Q-values using Bellman equation
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # If game continues, add discounted future reward
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update Q-value for the action that was taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Compute loss and update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()  # Return loss for monitoring


# Import numpy for array handling
import numpy as np