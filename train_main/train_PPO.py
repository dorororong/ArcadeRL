import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from env import Jumping_ball  # Jumping_ball 환경은 이전 코드에서 가져옵니다.

checkpoint_dir = './train/'
log_dir = './logs/'

# 사용자 정의 CNN 정의
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, observation_height=80, observation_width=80):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),  # Custom pooling to reduce height and width differently
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
        )

        # CNN 출력 크기 계산
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, observation_height, observation_width)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Linear layer to produce the desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.cnn(observations)
        return self.linear(x)



# 콜백 정의
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path,last_timesteps , verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.last_timesteps= last_timesteps

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            total_timesteps = self.model.num_timesteps + self.last_timesteps
            model_path = os.path.join(self.save_path, f'best_model_{total_timesteps}.zip')
            self.model.save(model_path)
        return True

if __name__ == '__main__':
    mode = 'test'


    if mode == 'train':

        # 환경 생성 및 래핑
        env = Jumping_ball()
        obs_height, obs_width = env.observation_height, env.observation_width
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=3)

        # 하이퍼파라미터 설정
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512,observation_height=obs_height,observation_width=obs_width)
        )

        try :
            models = os.listdir('train')
            lastmodel_name = "best_model_4220000.zip"
            timestep_last = int(lastmodel_name.split('_')[-1].split('.')[0])
            print(timestep_last)
            model = PPO.load(f'train/{lastmodel_name}', env=env, device="cpu")
        except:
            model = PPO(
                "CnnPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=2.5e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=4,
                clip_range=0.1,
                gamma=0.99,
                gae_lambda=0.95,
                policy_kwargs=policy_kwargs,
                device="cpu"
            )
            timestep_last = 0
            print("new model created")
        # 콜백 설정
        checkpoint_dir = './train/'
        log_dir = './logs/cleaned_game/'
        callback = TrainAndLoggingCallback(check_freq=5000, save_path=checkpoint_dir, last_timesteps=timestep_last)
        model.learn(total_timesteps=800000, callback=callback)


    else:
        def test_model(model_path, num_episodes=10, deterministic=True, render=False):
            # 환경 생성
            env = Jumping_ball()
            obs_height, obs_width = env.observation_height, env.observation_width
            print(obs_height, obs_width)
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=3)
            timestep_last = 0

            # 모델 로드
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=512,observation_height=obs_height,observation_width=obs_width)
            )
            model = PPO.load(model_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})

            # 평가
            all_rewards = []
            for episode in range(num_episodes):
                obs = env.reset()
                done = False
                total_reward = 0
                while not done:
                    if render:
                        env.render()
                    action, _states = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                all_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            average_reward = sum(all_rewards) / num_episodes
            print(f"Average Reward over {num_episodes} episodes: {average_reward}")

            # 환경 종료
            env.close()
    
        test_model(f'train/best_model_15000.zip', num_episodes=10, deterministic=True, render=False)


