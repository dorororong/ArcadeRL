# main.py
import argparse
from dueling import train_dueling
from vanilla_dqn import train_vanilla

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", choices=["dueling","vanilla"],
                        help="Which DQN variant to train")
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--buffer-size",      type=int, default=10000)
    parser.add_argument("--batch-size",       type=int, default=64)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--alpha",            type=float, default=0.6)
    parser.add_argument("--beta-start",       type=float, default=0.4)
    parser.add_argument("--beta-frames",      type=int,   default=20000)
    parser.add_argument("--eps-end",          type=float, default=0.05)
    parser.add_argument("--eps-decay",        type=int,   default=10000)
    parser.add_argument("--target-update",    type=int,   default=1000)
    args = parser.parse_args()

    if args.algo == "dueling":
        train_dueling(args)
    else:
        train_vanilla(args)
