#!/usr/bin/env python3
"""
Falling‑Duck PPO 학습/평가 스크립트
- 원시 Env → Monitor 래핑 → DummyVecEnv → VecFrameStack
- 텐서보드(logs/)에 평균 보상·에피소드 길이 자동 기록
"""
import os
import json
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from env import Env                             # 사용자 정의 Gym 환경
from PPO import create_or_load_model, TrainAndLoggingCallback
# Rainbow‑DQN 용 import (필요 없으면 삭제해도 됨)
from DQN import (
    create_or_load_model as create_or_load_model_dqn,
    TrainAndLoggingCallback as TrainAndLoggingCallbackDQN,
)
from network_light import MiniCNN               # 경량 CNN 추출기


# ────────────────────────────────────────────────────────────────
def make_env(game_name: str, fps: int = 7):
    """raw Env → Monitor 로 감싸서 반환하는 thunk"""
    def _thunk():
        env = Env(
            game_name=game_name,
            jump_key="space",
            fps=fps,
            debug=False,
        )
        return Monitor(env)        # ★ 반드시 원시 env를 먼저 Monitor 로 래핑
    return _thunk


# ────────────────────────────────────────────────────────────────
def main():
    # ─── 인자 파싱 ───────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--model", type=str, default="")      # 불러올 모델 이름
    parser.add_argument("--replay_buffer", type=str, default="")  # 불러올 리플레이 버퍼 이름
    parser.add_argument("--game_name", type=str, default="falling_duck")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "falling_duck_online.json"),
                        help="region 정의가 담긴 JSON 파일 경로")
    args = parser.parse_args()

    # ─── JSON으로부터 픽셀 크기 읽기 (필요 시) ───────────────
    with open(args.config, "r") as f:
        cfg = json.load(f)
    pix_w = cfg["regions"]["game"]["pixel_width"]
    pix_h = cfg["regions"]["game"]["pixel_height"]
    # Env 내부에서 observation_space shape=(n_stack, pix_h, pix_w) 사용

    # ─── VecEnv + FrameStack 구성 ────────────────────────────
    env = DummyVecEnv([make_env(args.game_name, fps=6)])
    env = VecFrameStack(env, n_stack=3)  

    # ─── PPO 네트워크 설정 ─────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=MiniCNN,
        features_extractor_kwargs=dict(features_dim=64)
    )

    # ─── 모델 생성 또는 불러오기 ────────────────────────────
    model, start_step = create_or_load_model_dqn(
        model_dir="./train/",
        log_dir="./logs/",
        model_name=args.model,
        env=env,
        policy_kwargs=policy_kwargs,
    )
    if args.replay_buffer:
        print(f"Loading replay buffer... {args.replay_buffer}")
        model.load_replay_buffer(os.path.join("./train/", args.replay_buffer))

    # print model parameters such as learning rate, batch size, etc.
    print("Model Parameters:")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"Batch Size: {model.batch_size}")
    print(f"Buffer Size: {model.buffer_size}")
    print(f"Gamma: {model.gamma}")
    print(f"train_freq: {model.train_freq}")


    # ─── 학습 / 평가 모드 ───────────────────────────────────
    if args.mode == "train":
        callback = TrainAndLoggingCallbackDQN(
            check_freq=5_000,
            save_path="./train/",
            last_timesteps=start_step,
        )
        model.learn(total_timesteps=args.timesteps, callback=callback, reset_num_timesteps=False)

        buffer_path = os.path.join("./train/", f"replay_{model.num_timesteps}.pkl")
        model.save_replay_buffer(buffer_path)
        print(f"Replay buffer saved to {buffer_path}")
        

    else:  # -------- 평가 모드 --------
        episodes = 10
        for ep in range(1, episodes + 1):
            obs = env.reset()
            done, total_reward = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            print(f"Episode {ep:02d} | reward = {total_reward}")
            time.sleep(1)


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
