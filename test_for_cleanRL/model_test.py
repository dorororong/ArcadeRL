# test_cartpole.py
import torch
import argparse
import numpy as np
import gymnasium as gym

from simple_mlp import DuelingMLP, MLPExtractor
from vanilla_dqn import QNet

def load_policy(algo: str, obs_space, action_space, device):
    """
    algo 에 따라 DuelingMLP 또는 QNet 을 인스턴스화 해 반환합니다.
    모델 파라미터 로드는 main() 에서 처리합니다.
    """
    if algo == "dueling":
        return DuelingMLP(obs_space, action_space).to(device)
    else:  # "vanilla"
        return QNet(obs_space, action_space).to(device)

def test(env_id, policy, episodes: int, render: bool, device):
    """
    주어진 policy 로 episodes 만큼 테스트하고,
    각 에피소드 리턴을 리스트로 반환합니다.
    """
    # 렌더 모드 지정
    render_mode = "human" if render else None
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    returns = []
    for ep in range(1, episodes+1):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                state_v = torch.tensor(obs, device=device).unsqueeze(0).float()
                action = policy(state_v).argmax(dim=1).item()
            obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            total_reward += reward
        returns.append(total_reward)
        print(f"Episode {ep:>2} | Return: {total_reward:.2f}")
    env.close()
    return returns

def main():
    parser = argparse.ArgumentParser(
        description="Test CartPole-v1 with either 'dueling' or 'vanilla' DQN model"
    )
    parser.add_argument(
        "--env-id", type=str, default="CartPole-v1",
        help="Gym environment ID"
    )
    parser.add_argument(
        "--model", choices=["dueling","vanilla"], required=True,
        help="Which model to test"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Optional path to the .pt file; if omitted, uses 'models/{model}.pt'"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Whether to render the environment"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 기본 모델 경로 결정
    model_path = args.model_path or f"models/{args.model}.pt"

    # 환경 생성 (for shape)
    dummy_env = gym.make(args.env_id)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()

    # 정책 네트워크 로드
    policy = load_policy(args.model, obs_space, act_space, device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"\nTesting '{args.model}' model from {model_path} for {args.episodes} episodes\n")
    returns = test(args.env_id, policy, args.episodes, args.render, device)
    avg = np.mean(returns)
    print(f"\nAverage return over {args.episodes} episodes: {avg:.2f}\n")

if __name__ == "__main__":
    main()
