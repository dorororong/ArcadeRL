import mss
import numpy as np
import cv2
import time
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import pydirectinput

from PIL import Image





class Jumping_ball(Env):
    def __init__(self,fps=10):
        super().__init__()
        self.game_region = {'top': 355, 'left': 1404, 'width': 450, 'height': 52}
        self.end_state_region = {'top': 186, 'left': 1308, 'width': 4, 'height': 4}
        self.A_button_coords = (1900, 200)
        self.action_space = Discrete(2)
        self.observation_height = int(self.game_region['height']/5)
        self.observation_width = int(self.game_region['width']/5)
        print(self.observation_height,self.observation_width)
        self.observation_space = Box(low=0, high=255, shape=(1,self.observation_height, self.observation_width), dtype=np.uint8)
        self.state = None
        self.target_fps = fps
        self.frame_duration = 1.0 / fps  # e.g., ~0.1667 for 6 fps
        self.last_step_time = None

    def capture_screen(self, region=None):
        with mss.mss() as sct:
            img = sct.grab(region)
            img = np.array(img, dtype=np.uint8)[:, :, 0]  # Assuming the first channel is sufficient
            return img

    def resize_image(self, image):
        # Since the image is already grayscale, skip color conversion
        resized_img = cv2.resize(image, (self.observation_width, self.observation_height), interpolation=cv2.INTER_AREA).astype(np.uint8)
        return resized_img[np.newaxis, :, :]

    def check_done(self):
        done = False
        # check color method if the area turns black, it means game over
        done_img = np.array(self.capture_screen(self.end_state_region))
        grey = np.squeeze(done_img)
        if np.mean(grey) < 0.2:
            done = True

        return done, grey

        pass

    def click_A_button(self,key=True):
        # enter "q" button to jump
        if key:
            pydirectinput.press('q')
        else:
            x,y=self.A_button_coords
            pydirectinput.click(x,y)

    def get_observation(self):
        raw = self.capture_screen(self.game_region)
        grey = self.resize_image(raw)
        return grey

    def step(self, action):

        start_time = time.time()

        truncated = False
        reward = 1

        if action == 1:
            self.click_A_button(key=True)


        done, grey = self.check_done()
        if done:
            reward = -5
            self.click_A_button(key=True)
        
        new_observation = self.get_observation()


        # ---------------------------------
        # Enforce a maximum FPS
        elapsed = time.time() - start_time
        remaining = self.frame_duration - elapsed
        if remaining > 0:
            time.sleep(remaining)
        # ---------------------------------

        info = {}
        return new_observation, reward, done, truncated, info

    def render(self):
        plt.imshow(np.squeeze(self.get_observation()), cmap='gray')
        plt.show()

    def reset(self, seed=None, options=None):
        # Seed the environment
        self.np_random, _ = seeding.np_random(seed)

        time.sleep(0.1)
        self.click_A_button(key=False)

        observation = self.get_observation()
        info = {}
        return observation, info


if __name__ == '__main__':
    env = Jumping_ball()
    obs = env.get_observation()
    plt.imshow(np.squeeze(obs), cmap='gray')
    print(obs)
    print(obs.shape)
    plt.show()
    done,end_img = env.check_done()
    print(end_img)
    plt.imshow(np.squeeze(end_img), cmap='gray')
    plt.show()
    print(done)
    env.click_A_button(key=False)