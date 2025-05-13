# vanilla_dqn.py
import random, math, os, logging
from collections import namedtuple

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from simple_mlp import MLPExtractor
from utils import make_log_dir

# 1. Q‐network (MLP + head)
class QNet(nn.Module):
    def __init__(self, obs_space, action_space, features_dim=128):
        super().__init__()
        self.net  = MLPExtractor(obs_space.shape[0], features_dim)
        self.head = nn.Linear(features_dim, action_space.n)
    def forward(self, x):
        x = self.net(x)
        return self.head(x)

# 2. Experience 타입
Experience = namedtuple("Experience", ["obs","action","reward","next_obs","done"])

# 3. Uniform Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, exp: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            # overwrite oldest
            self.buffer[self.pos] = exp
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Experience(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def train_vanilla(args):
    # ─ 로그 세팅 ───────────────────────────────
    log_dir = make_log_dir(algo_name="vanilla")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir,"train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir=os.path.join(log_dir,"tensorboard"))

    # ─ 환경·모델·버퍼 세팅 ──────────────────────
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net   = QNet(env.observation_space, env.action_space).to(device)
    tgt_net = QNet(env.observation_space, env.action_space).to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    # ε 스케줄
    eps_start, eps_end, eps_decay = 1.0, args.eps_end, args.eps_decay
    global_step, episode, ep_ret = 0, 0, 0.0
    obs, _ = env.reset(seed=0)

    # ─ 학습 루프 ────────────────────────────────
    while global_step < args.total_timesteps:
        # ε‐greedy 액션 선택
        eps = eps_end + (eps_start - eps_end) * math.exp(-1. * global_step / eps_decay)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                st = torch.tensor(obs, device=device).unsqueeze(0).float()
                a = q_net(st).argmax(dim=1).item()

        next_obs, r, done, trunc, _ = env.step(a)
        done_flag = done or trunc
        ep_ret += r

        # 버퍼에 저장
        buffer.add(Experience(obs, a, r, next_obs, done_flag))
        obs = next_obs
        global_step += 1

        # 학습 업데이트
        if len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            b_obs   = torch.tensor(batch.obs,   device=device).float()
            b_act   = torch.tensor(batch.action,device=device).long()
            b_r     = torch.tensor(batch.reward,device=device).float()
            b_next  = torch.tensor(batch.next_obs,device=device).float()
            b_done  = torch.tensor(batch.done,  device=device).float()

            # 현재 Q
            q_values = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            # Double DQN 타깃 계산
            with torch.no_grad():
                next_actions = q_net(b_next).argmax(1)
                next_q = tgt_net(b_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = b_r + args.gamma * (1 - b_done) * next_q

            # 손실 및 경사 하강
            loss = nn.functional.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            # 로깅
            logger.info(f"step={global_step} loss={loss.item():.4f} eps={eps:.3f}")
            writer.add_scalar("loss/td", loss.item(), global_step)
            writer.add_scalar("train/eps", eps, global_step)

        # 타깃 네트워크 하드 업데이트
        if global_step % args.target_update == 0:
            tgt_net.load_state_dict(q_net.state_dict())

        # 에피소드 종료 시
        if done_flag:
            episode += 1
            if episode % 50 == 0:
                print(f"[Ep {episode}], timestep {global_step} ret={ep_ret:.2f}")
            logger.info(f"[Ep {episode}] Return={ep_ret:.2f}")
            writer.add_scalar("episode/return", ep_ret, episode)
            ep_ret = 0.0
            obs, _ = env.reset()

    # ─ 모델 저장 ───────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save(q_net.state_dict(), os.path.join("models","vanilla.pt"))
    writer.close()
    print(f"Vanilla DQN done. logs → {log_dir}")
