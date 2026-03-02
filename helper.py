"""
Plotting Helper for Training Visualization

Displays real-time training progress with score plots.
Works both in Jupyter notebooks and regular Python scripts.
"""

import matplotlib.pyplot as plt


# Configure matplotlib for better display
plt.ion()  # Interactive mode


def plot(scores, mean_scores):
    """
    Plot training progress in real-time.
    
    Shows:
    - Individual game scores (blue line)
    - Running mean score (orange line)
    - Text labels for current values
    
    Args:
        scores: List of all game scores
        mean_scores: List of mean scores (calculated over last 100 games)
    
    Usage:
        scores = []
        mean_scores = []
        
        for game in range(1000):
            score = play_game()
            scores.append(score)
            mean_scores.append(np.mean(scores[-100:]))
            plot(scores, mean_scores)
    """
    plt.clf()  # Clear current figure
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Plot both lines
    plt.plot(scores, label='Score', color='#2196F3', alpha=0.7)
    plt.plot(mean_scores, label='Mean Score (Last 100)', 
             color='#FF9800', linewidth=2)
    
    # Set y-axis to start at 0
    plt.ylim(ymin=0)
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add text labels for latest values
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], 
                 f'{scores[-1]:.0f}', 
                 fontsize=10, 
                 verticalalignment='bottom')
    
    if len(mean_scores) > 0:
        plt.text(len(mean_scores) - 1, mean_scores[-1], 
                 f'{mean_scores[-1]:.1f}', 
                 fontsize=10, 
                 verticalalignment='top')
    
    # Update display
    plt.pause(0.001)  # Small pause to update the plot


def save_plot(scores, mean_scores, filename='training_plot.png'):
    """
    Save the final training plot to a file.
    
    Args:
        scores: List of all game scores
        mean_scores: List of mean scores
        filename: Output filename (default: 'training_plot.png')
    """
    plt.figure(figsize=(10, 6))
    plt.title('Final Training Results')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.plot(scores, label='Score', color='#2196F3', alpha=0.7)
    plt.plot(mean_scores, label='Mean Score (Last 100)', 
             color='#FF9800', linewidth=2)
    
    plt.ylim(ymin=0)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()