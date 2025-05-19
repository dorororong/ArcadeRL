# cartpole_env.py
import gymnasium as gym

def make_env():
    """
    Returns a new instance of CartPole-v1.
    """
    return gym.make("MountainCar-v0")



if __name__ == "__main__":
    # test the environment with human rendering
    env = gym.make("LunarLander-v2", render_mode="human")
    obs, _ = env.reset(seed=42)
    done = False
    for _ in range(100):
        action  =1
        obs, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        print(f"obs: {obs}, reward: {reward}, done: {done}")

    env.close()