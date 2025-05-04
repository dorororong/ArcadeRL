# main.py
#!/usr/bin/env python3
import os
import json
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env import Env
from PPO import create_or_load_model, TrainAndLoggingCallback
from network import CustomCNN
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--game_name", type=str, default="falling_duck")
    parser.add_argument("--config", type=str, default= os.path.join(os.path.dirname(__file__),"falling_duck_online.json"),
                        help="region 정의가 담긴 JSON 파일 경로")
    args = parser.parse_args()

    # ─── JSON 로부터 pixel 크기 읽기 ──────────────────────────
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    pix_w = cfg["regions"]["game"]["pixel_width"]
    pix_h = cfg["regions"]["game"]["pixel_height"]

    # ─── Env 생성 시 pixel 크기 넘겨주기 ─────────────────────
    #    (Env 내부에서 observation_space를
    #     spaces.Box(low=0, high=255, shape=(n_stack, pix_h, pix_w)) 등으로 설정)
    env0 = Env(
        game_name=args.game_name,
        jump_key="a",
        fps=6,
        debug=False
    )
    
    env = DummyVecEnv([lambda: env0])
    env = VecFrameStack(env, n_stack=3)

    # ─── PPO 에 전달할 Network 정의 ─────────────────────────
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            # features_dim 외에 추가 인자는 필요 없습니다.
            features_dim=256
        ),
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
            last_timesteps=start_step,
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
    else:
        # 평가 모드
        episodes = 10
        for ep in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            print(f"Episode {ep+1} reward= {total_reward}")
            time.sleep(1)
if __name__ == "__main__":
    main()