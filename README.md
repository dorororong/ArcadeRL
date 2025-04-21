# Jumping Ball RL Project

This project provides a complete pipeline for training and testing reinforcement learning agents on a browser-based jumping ball (or similar) game, using Gymnasium‑style environments, Stable‑Baselines3 PPO, and a custom CNN feature extractor.

---

## Reference
This project incorporates code and ideas from the MakeCode-Arcade_Electron repository by UnsignedArduino:

- **Original Repo**: [MakeCode-Arcade-to-Electron](https://github.com/UnsignedArduino/MakeCode-Arcade-to-Electron)
- **Modified files**: `main.py`, `app_build.py`

---

## Project Structure
```text
.
├── get_coord.py     # Screen‑region selector GUI → save regions + pixel sizes to JSON
├── env.py           # JumpingBallEnv: Gymnasium‑style environment
├── cnn.py           # CustomCNN: CNN feature extractor for PPO
├── ppo.py           # PPO helper: model creation/loading & checkpoint callback
├── main.py          # Entry point: train or test workflow via CLI
├── README.md        # This file
├── train/           # Checkpoints and saved models
└── logs/            # TensorBoard logs
```

### File Descriptions
- **get_coord.py**  
  - Launches a transparent overlay to drag‑select multiple regions on screen.  
  - For each region, reads pixel width/height inputs, saves all regions into a single JSON (with region coords + pixel sizes).  
  - Usage: `python get_coord.py` → Select & save → JSON file output.

- **env.py**  
  - Defines `JumpingBallEnv`, a Gymnasium `Env` capturing an on‑screen game region, resizing to the specified pixel dimensions, and mapping actions to key presses.  
  - Loads region configs from the JSON created by `get_coord.py`.

- **cnn.py**  
  - Implements `CustomCNN`, a Stable‑Baselines3 `BaseFeaturesExtractor` that applies convolution, pooling, and a final linear layer to image observations.

- **ppo.py**  
  - Provides `create_or_load_model()` for loading an existing PPO checkpoint or building a new model with specified hyperparameters and `CustomCNN`.  
  - Defines `TrainAndLoggingCallback` to save checkpoints every _N_ steps.

- **main.py**  
  - Parses CLI arguments (`--mode train|test`, `--timesteps`, `--model`).  
  - Wraps `JumpingBallEnv` in `DummyVecEnv` and `VecFrameStack`.  
  - Calls `create_or_load_model()` and either trains (`model.learn`) or evaluates the agent.

---

## Installation
1. Clone this repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   _requirements.txt_ should include:
   ```text
   gymnasium
   stable-baselines3[extra]
   torch
   mss
   opencv-python
   pillow
   pydirectinput
   tkinter
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

---

## Usage

1. **Select and save screen regions**  
   ```bash
   python get_coord.py
   ```
   - Drag to select one or more regions, enter each region’s name and pixel size, then click **저장 및 종료** to produce a `*.json` config.

2. **Train or test agent**  
   ```bash
   python main.py --mode train --timesteps 800000
   # or
   python main.py --mode test --model best_model_800000.zip
   ```

3. **Monitor training**  
   - Checkpoints will be saved under `train/`.  
   - TensorBoard logs are under `logs/`:
     ```bash
     tensorboard --logdir logs/
     ```

---

## Customization
- **Change CNN architecture**: edit `cnn.py`.  
- **Adjust PPO hyperparameters**: modify `ppo.py` or pass additional flags to `main.py`.  
- **Add new environments**: extend `env.py` or add new modules mapping other games.

---

Happy training! 🎮🤖

