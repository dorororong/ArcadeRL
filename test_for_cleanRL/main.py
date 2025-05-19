# main.py
import argparse
import random
import numpy as np
import torch    

# Import the correct training function
from dueling import train_dueling
# from vanilla_dqn import train_van     illa # Assuming you might have this; placeholder for now

# Placeholder for vanilla_dqn if not fully implemented for this specific request
def train_vanilla(args):
    print(f"Placeholder: Vanilla DQN training called with args: {args}")
    print("Note: This response focuses on Dueling DQN with CNN.")
    # If you have a vanilla_dqn.py that works with CNNs and your env,
    # you would import and call its training function here.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training for Arcade Game")
    
    # --- Algorithm Choice ---
    parser.add_argument("algo", choices=["dueling", "vanilla"], default="dueling", nargs='?',
                        help="Which DQN variant to train (default: dueling)")

    # --- General Training Hyperparameters ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total number of agent steps") # Increased
    parser.add_argument("--num-episodes", type=int, default=10000, help="Total number of episodes to run (alternative to timesteps)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer") # Often lower for CNNs
    parser.add_argument("--buffer-size", type=int, default=30_000, help="Max size of replay buffer") # Increased for images
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training") # Keep small for CNNs if memory is an issue
    parser.add_argument("--target-update", type=int, default=1000, help="Frequency (in agent steps) to update target network")
    parser.add_argument("--train-frequency", type=int, default=4, help="Frequency (in agent steps) to train the online network")

    # --- Epsilon-Greedy Exploration ---
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting value of epsilon")
    parser.add_argument("--eps-end", type=float, default=0.01, help="Minimum value of epsilon") # Lower for longer training
    parser.add_argument("--eps-decay", type=int, default=10_000, help="Number of steps over which to decay epsilon") # Slower decay

    # --- Prioritized Experience Replay (PER) ---
    parser.add_argument("--alpha", type=float, default=0.6, help="PER alpha (priority exponent)")
    parser.add_argument("--beta-start", type=float, default=0.4, help="PER beta (importance sampling exponent) starting value")
    parser.add_argument("--beta-frames", type=int, default=500_000, help="PER beta annealing frames (how many steps to reach beta=1.0)")

    # --- CNN Specific ---
    parser.add_argument("--cnn-features-dim", type=int, default=64, help="Output features dimension from CNN")
    parser.add_argument("--clip-grad-norm", type=float, default=10.0, help="Max norm for gradient clipping, None to disable")


    # --- Environment Specific ---
    # (game_name, json_file, config_dir are hardcoded in Env for now, but could be args)
    parser.add_argument("--env-fps", type=float, default=10, help="FPS for the game environment")
    parser.add_argument("--env-debug", action="store_true", help="Enable debug mode in environment (shows frames)")
    parser.add_argument("--env-save-frames", action="store_true", help="Save debug frames from environment")
    parser.add_argument("--env-initial-refresh", action="store_true", help="Perform an initial page refresh in environment")


    # --- Logging and Saving ---
    parser.add_argument("--log-dir-base", type=str, default="training_logs_jumping_ball", help="Base directory for logs")
    parser.add_argument("--log-interval", type=int, default=1000, help="Frequency (in global steps) to log training data")
    parser.add_argument("--save-interval", type=int, default=50, help="Frequency (in episodes) to save the model")


    args = parser.parse_args()

    # --- Set Seeds for Reproducibility ---
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            # Using deterministic algorithms can impact performance, enable if strict reproducibility is critical
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False


    # --- Dispatch to Training Function ---
    if args.algo == "dueling":
        print("Starting Dueling DQN with CNN training...")
        train_dueling(args)
    elif args.algo == "vanilla":
        print("Starting Vanilla DQN training...")
        train_vanilla(args) # This will call the placeholder
    else:
        print(f"Algorithm '{args.algo}' not recognized.")