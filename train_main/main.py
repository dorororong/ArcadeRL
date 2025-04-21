# main.py
#!/usr/bin/env python3

import os
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env import JumpingBallEnv
from PPO import create_or_load_model, TrainAndLoggingCallback
from network import CustomCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","test"], default="train")
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    # 환경 래핑
    env0 = JumpingBallEnv()  # JSON로 설정 로드하도록 수정 가능
    obs_h, obs_w = env0.observation_space.shape[1:]
    env = DummyVecEnv([lambda: env0])
    env = VecFrameStack(env, n_stack=3)

    # PPO 에 사용할 CNN 지정
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=512,
            observation_height=obs_h,
            observation_width=obs_w,
        )
    )

    # 모델 생성 또는 불러오기
    model, start_step = create_or_load_model(
        model_dir="./train/",
        log_dir="./logs/",
        model_name=args.model,
        env=env,
        policy_kwargs=policy_kwargs
    )

    if args.mode == "train":
        callback = TrainAndLoggingCallback(
            check_freq=5_000,
            save_path="./train/",
            last_timesteps=start_step
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
    else:
        # 평가 모드
        episodes = 10
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
            print(f"Episode {ep+1} reward= {total_reward}")


if __name__ == "__main__":
    main()