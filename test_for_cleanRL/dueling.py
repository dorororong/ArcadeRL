# dueling.py
import random, math, os, logging, time # Added time
from collections import namedtuple

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import your custom environment
# Make sure 'env_dx_gym_fast.py' is in the same directory or PYTHONPATH
try:
    from env import Env as CustomEnv
except ImportError:
    print("ERROR: Could not import 'Env' from 'env_dx_gym_fast.py'.")
    print("Please ensure your environment code is in a file named 'env_dx_gym_fast.py' and it's accessible.")
    exit()

# Import the Dueling CNN network
from dueling_cnn import DuelingCNN
from utils import make_log_dir # We'll create this utils.py file

# PER Buffer & Experience (Unchanged from your original)
Experience = namedtuple("Experience", ["obs","action","reward","next_obs","done"])
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity, self.alpha, self.eps = capacity, alpha, eps
        self.pos, self.buffer = 0, []
        self.prios = np.zeros((capacity,), dtype=np.float32) # For priorities

    def add(self, exp):
        max_prio = self.prios.max() if self.buffer else 1.0 # Use 1.0 if buffer is empty
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp # Overwrite oldest
        
        self.prios[self.pos] = max_prio # New experiences get max priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if not self.buffer: # Cannot sample if buffer is empty
            return [], Experience([], [], [], [], []), torch.tensor([])

        # Determine which priorities to use (either full buffer or up to current position)
        if len(self.buffer) == self.capacity:
            prios_to_sample = self.prios
        else:
            prios_to_sample = self.prios[:self.pos]

        # Calculate probabilities P(i) = p_i^alpha / sum(p_k^alpha)
        probs = prios_to_sample ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0: # Avoid division by zero if all priorities are zero (e.g., during initialization)
            # Fallback to uniform sampling if all priorities are zero
            probs = np.ones_like(prios_to_sample) / len(prios_to_sample)
        else:
            probs /= probs_sum
        
        # Ensure batch_size does not exceed current buffer size
        current_buffer_len = len(self.buffer)
        actual_batch_size = min(batch_size, current_buffer_len)

        indices = np.random.choice(current_buffer_len, actual_batch_size, p=probs, replace=False) # Sample without replacement if possible and batch small enough
        samples = [self.buffer[i] for i in indices]

        # Calculate importance sampling weights w_j = (N * P(j))^-beta / max(w_i)
        total_samples = len(self.buffer)
        weights = (total_samples * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize weights
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # Unpack samples into a batch
        batch = Experience(*zip(*samples))
        return indices, batch, weights_tensor

    def update(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.prios[idx] = abs(err) + self.eps # Add epsilon for non-zero priority


def train_dueling(args):
    # --- Logging Setup ---
    log_dir_base = args.log_dir_base if hasattr(args, 'log_dir_base') else "logs_jumping_ball"
    algo_name = f"duelingCNN_fps{args.env_fps}_lr{args.lr}"
    log_dir = make_log_dir(algo_name=algo_name, base_dir=log_dir_base)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    # Console handler for seeing logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    logger.info(f"Logging to: {log_dir}")
    logger.info(f"Arguments: {args}")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Environment Setup ---
    try:
        env = CustomEnv(
            fps=args.env_fps, # Pass FPS from args
            debug=args.env_debug,
            save_debug_frames=args.env_save_frames,
            initial_page_refresh=args.env_initial_refresh,
            # config_dir and json_file can also be args if needed
        )
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return # Exit if env fails

    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")


    # --- Network, Optimizer, Buffer Setup ---
    q_net   = DuelingCNN(env.observation_space, env.action_space, cnn_features_dim=args.cnn_features_dim).to(device)
    tgt_net = DuelingCNN(env.observation_space, env.action_space, cnn_features_dim=args.cnn_features_dim).to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    tgt_net.eval() # Target network in evaluation mode

    opt = optim.Adam(q_net.parameters(), lr=args.lr)
    buf = PrioritizedReplayBuffer(args.buffer_size, alpha=args.alpha)

    # --- Training Parameters ---
    eps_start, eps_end, eps_decay = args.eps_start, args.eps_end, args.eps_decay
    global_step = 0
    episode_count = 0
    
    # --- Training Loop ---
    try:
        start_time_training = time.time()
        for episode_count in range(1, args.num_episodes + 1): # Loop for a number of episodes
            current_episode_action_record = [] 
            obs, info = env.reset(seed=args.seed + episode_count if args.seed is not None else None)
            ep_ret = 0.0
            ep_len = 0
            episode_start_time = time.time()

            while True: # Loop for steps within an episode
                # Epsilon-greedy action selection
                eps = eps_end + (eps_start - eps_end) * math.exp(-1. * global_step / eps_decay)
                
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # Observation from env is (1, H, W) numpy array.
                        # MiniCNN expects (N, C, H, W) tensor.
                        # torch.tensor will create a tensor. Unsqueeze(0) adds batch dim if not already (1,C,H,W).
                        # obs is already (1,H,W), so tensor(obs) is (1,H,W). float() for type.
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)


                        # Ensure obs_tensor is (1, C, H, W) before passing to q_net
                        # Your env provides (1, H, W). This is correct for a single instance with 1 channel.
                        # If obs was (H,W), you'd need obs_tensor.unsqueeze(0).unsqueeze(0)
                        # If obs was (H,W,C), you'd need obs_tensor.permute(2,0,1).unsqueeze(0)
                        
                        q_values = q_net(obs_tensor)
                        action = q_values.argmax(dim=1).item()

                current_episode_action_record.append(action) # Record action for this episode
                next_obs, reward, done, truncated, info = env.step(action)
                ep_ret += reward
                ep_len += 1
                global_step += 1

                # Store experience in replay buffer
                # Ensure obs and next_obs are stored as numpy arrays (as they come from env)
                buf.add(Experience(obs, action, reward, next_obs, done or truncated)) # Store done if terminated or truncated
                obs = next_obs

                # --- Perform Network Update (Learning) ---
                if len(buf.buffer) >= args.batch_size and global_step % args.train_frequency == 0: # Train every N steps
                    beta = min(1.0, args.beta_start + global_step * (1.0 - args.beta_start) / args.beta_frames)
                    
                    idxs, batch, weights = buf.sample(args.batch_size, beta)
                    if len(idxs) == 0:
                        continue # Skip if buffer can't provide a sample

                    weights = weights.to(device)
                    
                    # Convert batch data to tensors
                    b_obs = torch.tensor(np.array(batch.obs), device=device).float() # np.array to stack correctly
                    b_act = torch.tensor(batch.action, device=device).long().unsqueeze(1) # Ensure (batch_size, 1) for gather
                    b_r   = torch.tensor(batch.reward, device=device).float().unsqueeze(1)
                    b_next= torch.tensor(np.array(batch.next_obs), device=device).float() # np.array to stack
                    b_done= torch.tensor(batch.done, device=device).float().unsqueeze(1)

                    # Q-values for current states and actions
                    current_q_values = q_net(b_obs).gather(1, b_act)

                    # Target Q-values (Double DQN)
                    with torch.no_grad():
                        next_actions = q_net(b_next).argmax(dim=1, keepdim=True)
                        next_q_values_target = tgt_net(b_next).gather(1, next_actions)
                        target_q_values = b_r + args.gamma * (1 - b_done) * next_q_values_target
                    
                    # TD error and loss
                    td_errors = target_q_values - current_q_values # Shape: (batch_size, 1)
                    loss = (td_errors.pow(2) * weights.unsqueeze(1)).mean() # Apply IS weights

                    # Optimization step
                    opt.zero_grad()
                    loss.backward()
                    if args.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(q_net.parameters(), args.clip_grad_norm)
                    opt.step()

                    # Update priorities in the buffer
                    buf.update(idxs, td_errors.squeeze().detach().cpu().numpy()) # Squeeze to (batch_size,)

                    # 변경: 타겟 네트워크 업데이트 시점에만 loss 로그 출력
                    if global_step % args.target_update == 0:
                        logger.info(f"Training Update - GlobalStep: {global_step}, Loss: {loss.item():.4f}")
                    writer.add_scalar("train/loss", loss.item(), global_step)

                # --- Target Network Update ---
                if global_step % args.target_update == 0:
                    tgt_net.load_state_dict(q_net.state_dict())
                    logger.info(f"GlobalStep: {global_step}: Target network updated.")

                if done or truncated:
                    break # End of episode

            # --- End of Episode Logging ---
            episode_duration = time.time() - episode_start_time
            steps_per_second = ep_len / episode_duration if episode_duration > 0 else 0
            
            writer.add_scalar("rollout/ep_reward", ep_ret, episode_count)
            writer.add_scalar("rollout/ep_length", ep_len, episode_count)
            writer.add_scalar("rollout/steps_per_second", steps_per_second, global_step)
            writer.add_scalar("rollout/epsilon", eps, global_step)

            logger.info(f"  Global Steps: {global_step}  Episode Return: {ep_ret:.2f}  Epsilon: {eps:.3f}")
            logger.info(f"  Action Record: {current_episode_action_record}")
            
            # Save model periodically
            if episode_count % args.save_interval == 0:
                model_save_path = os.path.join(log_dir, f"dueling_cnn_ep{episode_count}.pt")
                torch.save(q_net.state_dict(), model_save_path)
                logger.info(f"Model saved at episode {episode_count} to {model_save_path}")

            if global_step >= args.total_timesteps:
                logger.info(f"Reached total timesteps ({args.total_timesteps}). Stopping training.")
                break
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
    finally:
        # --- Save Final Model & Close Resources ---
        final_model_path = os.path.join(log_dir, "dueling_cnn_final.pt")
        torch.save(q_net.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        if 'env' in locals() and env: # Check if env was successfully initialized
            env.close()
        writer.close()
        
        total_training_time = time.time() - start_time_training
        logger.info(f"Total training time: {total_training_time / 3600:.2f} hours.")
        logger.info(f"Training finished. Logs and models are in: {log_dir}")