# dueling.py
import random, math, os, logging
from collections import namedtuple

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from simple_mlp import DuelingMLP
from utils import make_log_dir

# PER 버퍼 & 경험 타입
Experience = namedtuple("Experience", ["obs","action","reward","next_obs","done"])
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity, self.alpha, self.eps = capacity, alpha, eps
        self.pos, self.buffer = 0, []
        self.prios = np.zeros((capacity,), dtype=np.float32)
    def add(self, exp):
        max_prio = self.prios.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        self.prios[self.pos] = max_prio
        self.pos = (self.pos+1) % self.capacity
    def sample(self, batch_size, beta=0.4):
        prios = self.prios if len(self.buffer)==self.capacity else self.prios[:self.pos]
        probs = prios**self.alpha; probs /= probs.sum()
        idxs = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]
        total = len(self.buffer)
        weights = (total * probs[idxs])**(-beta); weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        batch = Experience(*zip(*samples))
        return idxs, batch, weights
    def update(self, idxs, errors):
        for i, e in zip(idxs, errors):
            self.prios[i] = abs(e) + self.eps

def train_dueling(args):
    # 로그 세팅
    log_dir = make_log_dir(algo_name="dueling")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir,"train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir=os.path.join(log_dir,"tensorboard"))

    # 환경·모델·버퍼
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net   = DuelingMLP(env.observation_space, env.action_space).to(device)
    tgt_net = DuelingMLP(env.observation_space, env.action_space).to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    opt     = optim.Adam(q_net.parameters(), lr=args.lr)
    buf     = PrioritizedReplayBuffer(args.buffer_size, alpha=args.alpha)

    # ε 스케줄 등
    eps_start, eps_end, eps_decay = 1.0, args.eps_end, args.eps_decay
    global_step, episode, ep_ret = 0, 0, 0.0
    obs, _ = env.reset(seed=0)

    # 학습 루프
    while global_step < args.total_timesteps:
        eps = eps_end + (eps_start-eps_end)*math.exp(-1.*global_step/eps_decay)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                st = torch.tensor(obs,device=device).unsqueeze(0).float()
                a = q_net(st).argmax(dim=1).item()

        next_obs, r, done, trunc, _ = env.step(a)
        done_flag = done or trunc
        ep_ret += r

        buf.add(Experience(obs,a,r,next_obs,done_flag))
        obs = next_obs; global_step += 1

        # 업데이트
        if len(buf.buffer) >= args.batch_size:
            beta = min(1.0, args.beta_start + global_step*(1.0-args.beta_start)/args.beta_frames)
            idxs, batch, weights = buf.sample(args.batch_size, beta)
            weights = weights.to(device)

            b_obs = torch.tensor(batch.obs,   device=device).float()
            b_act = torch.tensor(batch.action,device=device).long()
            b_r   = torch.tensor(batch.reward,device=device).float()
            b_next= torch.tensor(batch.next_obs,device=device).float()
            b_done= torch.tensor(batch.done,  device=device).float()

            qv = q_net(b_obs).gather(1,b_act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                na = q_net(b_next).argmax(1)
                nq = tgt_net(b_next).gather(1,na.unsqueeze(1)).squeeze(1)
                target = b_r + args.gamma*(1-b_done)*nq

            td = qv - target
            loss = (td.pow(2) * weights).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(),10.0)
            opt.step()
            buf.update(idxs, td.detach().cpu().numpy())

            # 로깅
            logger.info(f"step={global_step} loss={loss.item():.4f} eps={eps:.3f}")
            writer.add_scalar("loss/td", loss.item(), global_step)
            writer.add_scalar("train/eps", eps, global_step)

        if global_step % args.target_update == 0:
            tgt_net.load_state_dict(q_net.state_dict())

        if done_flag:
            episode += 1
            if episode % 100 == 0:
                print(f"[Ep {episode}] ret={ep_ret:.2f}")
            logger.info(f"[Ep {episode}] ret={ep_ret:.2f}")
            writer.add_scalar("episode/return", ep_ret, episode)
            ep_ret = 0.0
            obs, _ = env.reset()

    # 저장
    os.makedirs("models", exist_ok=True)
    torch.save(q_net.state_dict(), os.path.join("models","dueling.pt"))
    writer.close()
    print(f"Dueling done. logs → {log_dir}")
