#!/usr/bin/env python3
"""
High-speed Gymnasium Environment for screen-capture games using dxcam only.

Optimized grab: single full-frame grab → slice game/end regions.
Debug: real-time 윈도우 표시 + optional 파일 저장.
"""

import os
import json
import time
import traceback

import dxcam
import numpy as np
import cv2
import pydirectinput
from gymnasium import Env as GymEnv
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding

class Env(GymEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        game_name: str = "jumping_ball",
        jump_key: str = "space",
        fps: float = 10,
        config_dir: str = "train_main",
        end_check_every: int = 1,
        json_file: str = "jumping_ball_online",
        debug: bool = True,              # 디버깅 모드 플래그
        save_debug_frames: bool = True, # 프레임 저장 플래그
        debug_dir: str = "debug_frames", # 저장 폴더
        initial_page_refresh = False
    ):
        super().__init__()
        # 설정 로드
        cfg_path = os.path.join(config_dir, f"{json_file}.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        gr = cfg["regions"]["game"]
        er = cfg["regions"]["end"]
        self._last_page_reset = time.time()

        # 디버깅 세팅
        self.debug = debug
        self.save_debug = save_debug_frames
        if self.save_debug and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        self._debug_dir = debug_dir

        # 액션/관측 스페이스
        self.action_space = Discrete(2)
        self.pixel_w = gr["pixel_width"]
        self.pixel_h = gr["pixel_height"]
        self.observation_space = Box(0, 255, (1, self.pixel_h, self.pixel_w), np.uint8)

        # 액션 및 보상 기록
        self.action_record = []
        self.eps_reward = 0.0

        # DXCam 초기화
        self._cam = dxcam.create(output_idx=0, output_color="GRAY")
        if self._cam is None:
            raise RuntimeError("dxcam.create returned None")

        # 지역 좌표
        l, t, w, h = gr["left"], gr["top"], gr["width"], gr["height"]
        self._game_rect = (l, t, l + w, t + h)
        l2, t2, w2, h2 = er["left"], er["top"], er["width"], er["height"]
        self._end_rect  = (l2, t2, l2 + w2, t2 + h2)

        # 기타 런타임 설정
        pydirectinput.PAUSE = 0
        self.jump_key    = jump_key
        self._frame_dur  = 1.0 / fps if fps > 0 else None
        self._end_every  = max(1, end_check_every)
        self._frame_count = 0
        self.np_random, _ = seeding.np_random(None)


        # 1) game, end 두 사각형을 한 번에 덮는 공통 bbox
        gx1, gy1, gx2, gy2 = self._game_rect          # (l,t,r,b)
        ex1, ey1, ex2, ey2 = self._end_rect
        bx1, by1 = min(gx1, ex1), min(gy1, ey1)
        bx2, by2 = max(gx2, ex2), max(gy2, ey2)
        self._bbox = (bx1, by1, bx2, by2)              # <- grab(region=...)

        # 2) bbox 내부에서 game/end 의 상대 좌표 계산
        self._rel_game = (gx1-bx1, gy1-by1, gx2-bx1, gy2-by1)
        self._rel_end  = (ex1-bx1, ey1-by1, ex2-bx1, ey2-by1)

        # 환경 설정시 페이지 리셋 
        if initial_page_refresh:
            print("Initial reset...")
            self.page_refresh()

    def _grab_rois(self):
        """한 번의 grab 으로 game/end ROI 를 동시에 리턴."""
        full = self._cam.grab(region=self._bbox) 
        if full is None:
            return None, None
        # unpack
        lg,tg,rg,bg = self._rel_game
        le,te,re,be = self._rel_end
        roi_game = full[tg:bg, lg:rg]
        roi_end  = full[te:be, le:re]
        return roi_game, roi_end

    def _process_game(self, roi_game):
        """게임 프레임→관측값 (1,h,w) 반환."""
        if roi_game is None:
            return np.zeros((1, self.pixel_h, self.pixel_w), np.uint8)
        gray = roi_game 
        if gray.shape != (self.pixel_h, self.pixel_w): 
            gray = cv2.resize(gray, (self.pixel_w, self.pixel_h),
                              interpolation=cv2.INTER_NEAREST)
            
        # 디버깅용 프레임 저장
        if self.debug and self.save_debug:
            self._frame_count += 1
            cv2.imwrite(os.path.join(self._debug_dir, f"frame_{self._frame_count:04d}.png"), gray)
            cv2.imshow("Game Frame", gray)
            cv2.waitKey(1)
        return gray[np.newaxis, ...]

    def _process_end(self, roi_end):
        """종료 여부 판단: all-black 이면 종료."""
        if roi_end is None:
            return False
        gray = roi_end[:, :, 0]
        return np.all(gray == 0)

    def step(self, action: int):
        start = time.perf_counter()

        if action == 1:
            pydirectinput.press(self.jump_key)

        # # 2) 한 번만 grab → 두 ROI 분리
        roi_game, roi_end = self._grab_rois() 

        # 3) 관측 & 종료판정 & 보상
        obs   = self._process_game(roi_game)
        done = self._process_end(roi_end) 



        reward = -1.0 if done else 0.1
        self.eps_reward += reward
        self.action_record.append(action)

        # 4) FPS 캡
        if self._frame_dur:
            elapsed = time.perf_counter() - start
            to_sleep = self._frame_dur - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)


        # ─── 1시간 경과 시 페이지 리셋 ───────────────────────────
        now = time.time()
        if now - self._last_page_reset >= 2000:
            print("1시간 경과, 페이지 새로고침 실행")
            self.page_refresh()
            self._last_page_reset = now

        if done:
            print("action record:", self.action_record, f"eps_reward: {self.eps_reward:.2f}")

        return obs, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        print("Reset the environment...")
        time.sleep(1)
        super().reset(seed=seed)
        self.action_record = []
        self.eps_reward = 0.0
        self.np_random, _ = seeding.np_random(seed)
        self._frame_count = 0
        # 클릭으로 재시작
        l, t, r, b = self._game_rect
        cx, cy = (l + r)//2, (t + b)//2
        pydirectinput.click(x=cx, y=cy)
        time.sleep(0.1)
        # 초기 관측
        roi_game, _ = self._grab_rois()
        obs = self._process_game(roi_game)
        pydirectinput.press(self.jump_key)
        time.sleep(0.1) 
        return obs, {}

    def render(self, mode="human"):
        # env.step() 내부에서 이미 디버깅 윈도우로 보여주므로,
        # 따로 구현하지 않아도 됩니다.
        return

    def close(self):
        print("Closing env...")
        try:
            self._cam.stop()
        except:
            pass
        cv2.destroyAllWindows()

    def page_refresh(self):
        """게임 페이지를 리셋합니다."""
        l, t, r, b = self._game_rect
        cx, cy = (l + r)//2, (t + b)//2
        pydirectinput.click(x=cx, y=cy)
        time.sleep(1)
        pydirectinput.press("f5")
        time.sleep(10)
        pydirectinput.click(x=370, y=890)
        time.sleep(1)

if __name__ == "__main__":
    # simple smoke test
    env = None
    try:
        env = Env(
            fps=7,
            debug=True,
            save_debug_frames=True,
            debug_dir="debug_frames"
        )
        obs, _ = env.reset()

        episodes = 5  # 테스트할 에피소드 수
        for ep in range(episodes):
            action_record =[]
            start_time = time.time()  # 에피소드 시작 시간 기록
            step_count = 0  # 스텝 수 초기화
            total_reward = 0  # 총 보상 초기화
            done = False

            print(f"Starting Episode {ep + 1}...")
            while not done:
                action = env.action_space.sample()
                obs, reward, done, aa, bb = env.step(action)
                print(f"Step {step_count + 1}: Action: {action}, Reward: {reward}")
                print(f"Observation shape: {obs.shape}")
                step_count += 1
                total_reward += reward
                action_record.append(action)

            end_time = time.time()  # 에피소드 종료 시간 기록
            elapsed_time = end_time - start_time  # 경과 시간 계산
            fps = step_count / elapsed_time if elapsed_time > 0 else 0  # FPS 계산

            print(f"Episode {ep + 1} finished:")
            print(f"  Total reward: {total_reward}")
            print(f"  Steps: {step_count}")
            print(f"  Elapsed time: {elapsed_time:.2f} seconds")
            print(f"  FPS: {fps:.2f}")
            print(f"  Action record: {env.action_record}")
            print(f" eps_reward: {env.eps_reward:.2f}")

            # 에피소드 종료 후 환경 초기화
            obs, cc = env.reset()
            print("Environment reset for next episode.")
            print("reset obs:", obs.shape)  
            print("reset cc:", cc)

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
    finally:
        if env:
            env.close()
