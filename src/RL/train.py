import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import random
import importlib
from py4j.java_gateway import JavaGateway, CallbackServerParameters
from src.RL.params import *


def parse_args():
    parser = argparse.ArgumentParser("RL for Tetris")
    parser.add_argument("--alg", type=str, default="", help="reinforcement learning algorithm")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--num-games", type=int, default=1000, help="number of games")
    parser.add_argument("--save-dir", type=str, default="", help="directory to save policy and plots")
    args = parser.parse_args()
    return args


class PythonListener(object):

    def __init__(self, gateway, policy, replay_buffer):
        self.gateway = gateway
        self.policy = policy
        self.replay_buffer = replay_buffer

    def notify(self, state):
        print(state)
        action = self.policy.take_action(state)
        is_end = state[-1]
        self.gateway.entry_point.takeAction(action)
        reward = calculate_reward(self.replay_buffer.state_list[-1], state, is_end)
        self.replay_buffer.add(state, action, reward, is_end)
        return "A Return Value"

    class Java:
        implements = ["org.py4j.smallbench.BenchListener"]


class HeuristicPolicy:
    def __init__(self, args):
        self.args = args

    def take_action(self, state):
        field = np.array(state[0])
        next_piece_idx = state[1]
        is_end = state[2]
        w = pWidth[next_piece_idx][0]
        action = [0, 0]

        # top_list = get_top(field)
        # min_index = top_list.index(min(top_list))
        # action = [min_index, 0]

        return action


def get_top(field):
    h, w = field.shape
    top_list = []
    for j in range(w):
        top_i = 0
        for i in range(h):
            if field[i][j]:
                top_i = i
        top_list.append(top_i)
    return top_list



# class Policy:
#     def __init__(self, args):
#         # module = importlib.import_module(module_name)
#         algorithm = getattr(self, args.alg)
#         return
#
#     def take_action(self, state):
#         action = None
#         return action
#
#     def learn(self, replay_buffer):
#         return


class ReplayBuffer:
    def __init__(self):
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.is_end_list = []
        self.count = 0
        self.all_list = []

    def add(self, state, action, reward, is_end):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.is_end_list.append(is_end)

    def sample(self, num_samples):
        samples = random.sample(self.all_list, num_samples)
        return samples

    def post_process(self):
        self.all_list = zip(self.state_list, self.action_list, self.reward_list, self.is_end_list)
        return

    def get_average_reward(self):
        return 0


class Logger:
    def __init__(self, args):
        self.args = args
        self.reward_list = []

    def add(self, reward):
        self.reward_list.append(reward)

    def plot(self):
        plt.plot(self.reward_list)
        plt.savefig(os.path.join(self.args.save_dir, "reward.png"))


def calculate_reward(state, next_state, is_end):
    reward = 0
    return reward


def collect_data(policy, num_games=1):
    replay_buffer = ReplayBuffer()

    for game_index in range(num_games):
        gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
        listener = PythonListener(gateway)
        gateway.entry_point.registerBenchListener(listener)
        gateway.entry_point.startGames(1)

    return replay_buffer


# def main():
#     args = parse_args()
#     logger = Logger(args)
#     env = Environment()
#     policy = Policy(args)
#     for episode in range(args.num_episodes):
#         replay_buffer = collect_data(env, policy, num_games=args.num_games)
#         policy.learn(replay_buffer)
#         logger.add(replay_buffer.get_average_reward())
#     logger.plot()


def test_heuristic():
    args = parse_args()
    policy = HeuristicPolicy(args)
    replay_buffer = collect_data(policy, num_games=1)
    print(replay_buffer.reward_list)


if __name__ == "__main__":
    test_heuristic()
