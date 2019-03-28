import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from train import parse_args


def load_data(args, exp_name):
    with open(os.path.join(args.save_dir, exp_name, 'q.pkl'), 'rb') as f:
        q_list = pickle.load(f)
    with open(os.path.join(args.save_dir, exp_name, 'validation_reward.pkl'), 'rb') as f:
        reward_validation_list = pickle.load(f)
    with open(os.path.join(args.save_dir, exp_name, 'reward.pkl'), 'rb') as f:
        reward_list = pickle.load(f)
    with open(os.path.join(args.save_dir, exp_name, 'cleared.pkl'), 'rb') as f:
        cleared_list = pickle.load(f)

    return q_list, reward_validation_list, reward_list, cleared_list


if __name__ == "__main__":
    args = parse_args()

    for exp_name in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]:
        q_list, reward_validation_list, reward_list, cleared_list = args.load_data(args, exp_name)
        plt.figure("reward plot")
        plt.plot(reward_validation_list)

        plt.figure("cleared plot")
        plt.plot(cleared_list)

    plt.figure("reward plot")
    plt.savefig(os.path.join(args.save_dir, "reward.png"))
    plt.figure("cleared plot")
    plt.savefig(os.path.join(args.save_dir, "cleared.png"))
