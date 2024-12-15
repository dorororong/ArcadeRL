# Reference
This project incorporates code from the MakeCode-Arcade_Electron repository by UnsignedArduino. Below are the specific files that have been modified to suit local settings:
- main.py
- app_build.py
Original Repo : [MakeCode-Arcade_Electron] https://github.com/UnsignedArduino/MakeCode-Arcade-to-Electron



## train_main.py
# env.py
Implements a Gymnasium-like environment for offline reinforcement learning (RL) algorithms.

## get_coord.py
1. Captures a screenshot of the entire screen and displays a new window filled with the captured image.
2. Allows users to click twice to select the top-left and bottom-right points, defining the observation region.
3. Outputs the coordinates as "top, left, width, height," which should be copied and pasted into the game_region variable in env.py.

## train_PPO.py
- Trains and tests the Proximal Policy Optimization (PPO) algorithm using stable-baselines3 and a custom Convolutional Neural Network (CNN).
- Allows modification of the CNN architecture based on observations from the environment.
- Utilizes a vectorized environment with three stacked observations to enhance data capture.
