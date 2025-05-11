#DQN.py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, last_timesteps=0, verbose=1, save_buffer=True):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.last_timesteps = last_timesteps
        self.save_buffer = save_buffer

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            total = self.model.num_timesteps + self.last_timesteps
            self.model.save(os.path.join(self.save_path, f"best_model_{total}.zip"))
            if self.save_buffer:
                buf_path = os.path.join(self.save_path, f"replay_{total}.pkl")
                self.model.save_replay_buffer(buf_path)
        return True


def create_or_load_model(model_dir, log_dir, model_name, env, policy_kwargs):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if model_name:
        path = os.path.join(model_dir, model_name)
        model = DQN.load(path, env=env, policy_kwargs=policy_kwargs, device="cpu")
        step = int(model_name.split('_')[-1].split('.')[0])
    else:
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=2e-4,
            buffer_size=30_000,
            learning_starts=3_000,
            train_freq=(6, "step"),
            batch_size=32,
            gamma=0.99,
            max_grad_norm=10,
            target_update_interval=2_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.2,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=1,
        )
        step = 0

    return model, step
