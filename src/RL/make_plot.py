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
        if exp_name == 'exp3':
            cleared_list = list((np.array(reward_validation_list)/5))
        else:
            cleared_list = list((np.array(reward_validation_list)-10.0)/7.5)
    

    with open(os.path.join(args.save_dir, exp_name, 'reward.pkl'), 'rb') as f:
        reward_list = pickle.load(f)


    return q_list, reward_validation_list, reward_list, cleared_list


if __name__ == "__main__":
    args = parse_args()

    for exp_name in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]:
        q_list, reward_validation_list, reward_list, cleared_list = load_data(args, exp_name)
        # plt.figure("reward plot")
        # plt.plot(reward_validation_list)
        avg_cleared = []
        for i in range(int(len(cleared_list)/10)):
            avg_cleared.append(sum(cleared_list[i*10:i*10 + 10])/10.0)

        plt.figure("cleared plot")
        plt.plot(avg_cleared[:150])

    plt.legend(["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"])
    plt.xlabel("Number of games (in unit of 1000)")
    plt.ylabel("Average number of rows cleared")
    plt.title("Average number of rows cleared v.s. Number of games")


    plt.figure("cleared plot")
    plt.savefig(os.path.join(args.save_dir, "cleared.png"))
