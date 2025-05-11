# PPO.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, last_timesteps=0, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.last_timesteps = last_timesteps

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            total = self.model.num_timesteps + self.last_timesteps
            path = os.path.join(self.save_path, f"best_model_{total}.zip")
            self.model.save(path)
        return True


def create_or_load_model(model_dir, log_dir, model_name, env, policy_kwargs):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if model_name:
        path = os.path.join(model_dir, model_name)
        model = PPO.load(path, env=env, policy_kwargs=policy_kwargs, device="cpu")
        step = int(model_name.split('_')[-1].split('.')[0])
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=6e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=3,
            ent_coef=0.005,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs,
            device="cpu"
        )
        step = 0
    return model, step