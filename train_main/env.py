import mss
import numpy as np
import cv2
import time
import pydirectinput
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding

import matplotlib.pyplot as plt
import json


class JumpingBallEnv(Env):
    """
    A Gymnasium environment for a jump-based game (e.g., the Chrome Dino/Jumping Ball).
    Captures a region of the screen, downsamples to a fixed size, and sends "jump" actions.
    """
    metadata = {"render_modes": ["human"]}

    

    def __init__(
        self,
        jump_key: str = 'q',
        fps: int = 10,
        ):

        json_name = "jumping_ball"
        json_path = f"train_main/{json_name}.json"
        game_region_name = "game"
        end_region_name = "end"

        with open(json_path, "r") as f:
            cfg=json.load(f)

        self.game_region = cfg["regions"][game_region_name]
        self.end_region = cfg["regions"][end_region_name]

        super().__init__()
        # Screen regions: dict with keys top, left, width, height
        self.action_space = Discrete(2)
        # Observation: single-channel, channels-first
        self.pixel_w, self.pixel_h = self.game_region["pixel_width"], self.game_region["pixel_height"]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(1, self.pixel_h, self.pixel_w),
            dtype=np.uint8,
        )
        self.jump_key = "a"
        self.frame_duration = 1.0 / fps
        self.np_random, _ = seeding.np_random(None)

    def capture_screen(self, region: dict) -> np.ndarray:
        """
        Capture a grayscale screenshot of the given region.
        Returns a 2D NumPy array (height, width).
        """
        with mss.mss() as sct:
            img = sct.grab(region)
            arr = np.array(img)[:, :, 0]
            return arr

    def resize_and_gray(self, img: np.ndarray) -> np.ndarray:
        """
        Resize the raw grayscale image to (pixel_w, pixel_h) and add channel axis.
        Returns shape (1, pixel_h, pixel_w).
        """
        resized = cv2.resize(
            img,
            (self.pixel_w, self.pixel_h),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)
        return resized[np.newaxis, ...]

    def check_done(self) -> bool:
        """
        Determine if the game is over by inspecting the end_region.
        Returns True if mean pixel value is below a threshold.
        """
        done_img = self.capture_screen(self.end_region)
        return np.mean(done_img) < 10  # threshold, adjust as needed

    def click_jump(self) -> None:
        """
        Send the configured key press to perform a jump.
        """
        pydirectinput.press(self.jump_key)

    def get_observation(self) -> np.ndarray:
        """
        Grab the game region and return the processed observation.
        """
        raw = self.capture_screen(self.game_region)
        return self.resize_and_gray(raw)

    def step(self, action: int):  # type: ignore
        """
        Perform action (0 or 1), wait to enforce FPS, and return
        (observation, reward, done, truncated, info).
        """
        t0 = time.time()
        reward = 1.0
        if action == 1:
            self.click_jump()

        done = self.check_done()
        if done:
            reward = -5.0
            # optionally restart
            self.click_jump()

        obs = self.get_observation()
        # FPS cap
        dt = time.time() - t0
        if dt < self.frame_duration:
            time.sleep(self.frame_duration - dt)

        return obs, reward, done, False, {}

    def reset(self, *, seed=None, options=None):  # type: ignore
        """
        Reset the environment: seed RNG and perform an initial click if needed.
        Returns initial observation and empty info.
        """
        self.np_random, _ = seeding.np_random(seed)
        time.sleep(0.1)
        # initial click to start game (optional)
        pydirectinput.click(x=self.game_region['left'], y=self.game_region['top'])
        obs = self.get_observation()
        return obs, {}

    def render(self):
        """
        Optional: show the current observation via OpenCV or matplotlib.
        """
        import matplotlib.pyplot as plt
        plt.imshow(np.squeeze(self.get_observation()), cmap='gray')
        plt.show()



if __name__ == '__main__':
    env = JumpingBallEnv(Env)
    obs = env.get_observation()
    plt.imshow(np.squeeze(obs), cmap='gray')
    plt.show()
    time.sleep(1)
    env.click_jump()
    time.sleep(1)
    obs = env.get_observation()
    plt.imshow(np.squeeze(obs), cmap='gray')
    plt.show()
