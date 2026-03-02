"""
QUICK START GUIDE - Endless Runner AI
======================================

This guide gets you training your AI in 3 minutes!
"""

# Step 1: Check GPU Availability
# ================================
import torch

print("="*60)
print("SYSTEM CHECK")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✅ GPU training will be used automatically!")
else:
    print("ℹ️  Training will use CPU (slower but works fine)")

print("\n")

# Step 2: Test the Game
# ======================
print("="*60)
print("STEP 1: Test the Game")
print("="*60)
print("Run this command to play the game manually:")
print("  python game.py")
print("\nPress SPACE to jump, ESC to quit")
print("\n")

# Step 3: Start Training
# =======================
print("="*60)
print("STEP 2: Start Training")
print("="*60)
print("Run this command to start AI training:")
print("  python agent.py")
print("\nWhat to expect:")
print("- A window opens showing the game")
print("- AI plays automatically")
print("- Plot updates in real-time showing progress")
print("- Console shows stats every game")
print("\nTraining will look like this:")
print("  Game:   10 | Score:   45 | Best:   89 | Mean:   52.3")
print("  Game:   50 | Score:  120 | Best:  150 | Mean:   78.5")
print("  Game:  100 | Score:  200 | Best:  250 | Mean:  125.8")
print("\n")

# Step 4: Faster Training (Optional)
# ===================================
print("="*60)
print("STEP 3: Speed Up Training (Optional)")
print("="*60)
print("For MUCH faster training, edit agent.py:")
print("\nLine ~200, change:")
print("  game = Game(use_display=True)")
print("to:")
print("  game = Game(use_display=False)")
print("\nLine ~225, comment out:")
print("  # game.draw()")
print("  # game.clock.tick(60)")
print("\nThis runs 10x faster! (no graphics)")
print("\n")

# Step 5: Monitor Progress
# =========================
print("="*60)
print("MONITORING YOUR TRAINING")
print("="*60)
print("\nGood signs (AI is learning):")
print("  ✅ Best score increasing over time")
print("  ✅ Mean score trending upward")
print("  ✅ Epsilon decreasing to 0")
print("  ✅ Memory filling up")
print("\nBad signs (something wrong):")
print("  ❌ Scores stuck at same low value")
print("  ❌ Mean score not improving after 100+ games")
print("\nIf stuck, try:")
print("  - Train longer (500-1000 games)")
print("  - Adjust learning rate in agent.py (try 0.0001)")
print("  - Increase hidden layer size (try 512)")
print("\n")

# Step 6: Save and Test
# ======================
print("="*60)
print("SAVING & TESTING YOUR MODEL")
print("="*60)
print("\nYour best model auto-saves to:")
print("  ./model/best_model.pth")
print("\nTo stop training:")
print("  Press Ctrl+C (progress is saved)")
print("\nTo test your trained model:")
print("  1. Load the model in a new script")
print("  2. Set epsilon = 0 (no randomness)")
print("  3. Run with use_display=True to watch")
print("\n")

# Quick Tips
# ==========
print("="*60)
print("QUICK TIPS")
print("="*60)
print("• Let it train for at least 100-200 games")
print("• Early games will be bad (agent is learning)")
print("• Score should improve around game 50-100")
print("• GPU training is automatic if available")
print("• Plot shows progress visually")
print("• Best model saves automatically")
print("\n")

print("="*60)
print("READY TO START!")
print("="*60)
print("\n1. Run: python game.py (test manually)")
print("2. Run: python agent.py (start training)")
print("3. Watch it learn!")
print("\nGood luck! 🚀\n")
print("="*60)
