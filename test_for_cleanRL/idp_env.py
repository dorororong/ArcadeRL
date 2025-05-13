# cartpole_env.py
import gymnasium as gym

def make_env():
    """
    Returns a new instance of CartPole-v1.
    """
    return gym.make("CartPole-v1")