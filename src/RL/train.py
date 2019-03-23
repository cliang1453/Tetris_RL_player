import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import random
import importlib
from py4j.java_gateway import JavaGateway, CallbackServerParameters
from params import *
import math
from model import QFunc
import torch
import torch.optim as optim
from torch.autograd import Variable
from termcolor import colored
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("RL for Tetris")
    parser.add_argument("--alg", type=str, default="", help="reinforcement learning algorithm")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.95, help="alpha")
    parser.add_argument("--eps", type=float, default=0.01, help="eps")
    parser.add_argument("--epsilon-g", type=float, default=0.3, help="epsilon greedy")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--num-games", type=int, default=1000, help="number of games")
    parser.add_argument("--batch-size", type=int, default=32, help="number of games")
    parser.add_argument("--save-dir", type=str, default="log", help="directory to save policy and plots")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--max-capacity", type=int, default=1000, help="maximum capacity of replay buffer")

    args = parser.parse_args()
    return args


"""
state: [ndarray(rows * cols), next_piecee]

"""


class PythonListener(object):

    def __init__(self, args, gateway, policy, replay_buffer):
        self.args = args
        self.gateway = gateway
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.game_count = 0

    def is_valid(self):
        if self.game_count % 10 == 0:
            return True
        else:
            return False

    def notify(self, next_piece, field, rows_cleared, is_end):
        print(colored("=" * 40 + str(self.game_count) + "=" * 40, 'red'))
        # format state
        print("\tnext_piece\t", next_piece)
        print("\tis_end\t", is_end)
        field = list(field)
        field = [list(row) for row in field]
        field = field + [[0] * cols for _ in range(4)]  # add 4 rows on top
        field = np.array(field)
        print("\tfield", field[::-1])
        field = field > 0
        state = [field, next_piece]

        # select action
        if is_end:
            action = [0, 0]
        else:
            action = self.policy.take_action(state, is_valid=self.is_valid())

        if not self.is_valid():
            # store transition
            print("\t adding to replay buffer...")
            self.replay_buffer.add(state, action, rows_cleared, is_end)

            if self.replay_buffer.get_size() % 20 == 0:
                self.replay_buffer.visualize_replaybuffer()

            # sample and update
            if len(self.replay_buffer.valid_idx_list) > self.args.batch_size:
                print("\tsampling...")
                samples = self.replay_buffer.sample(self.args.batch_size)
                self.policy.learn(samples)

        # run simulation step
        print("\t calling java with action", action)
        if is_end:
            self.game_count += 1
            if self.is_valid():
                self.policy.save_params()

        self.gateway.entry_point.takeAction(int(action[0]), int(action[1]))

    class Java:
        implements = ["org.py4j.smallbench.BenchListener"]


class HeuristicPolicy:
    def __init__(self, args):
        self.args = args

    def take_action(self, state):
        action = [0, random.randint(1, 5)]
        return action


def get_top(board):
    top_board = np.zeros(cols)
    for row_id in range(rows):
        row = board[row_id]
        filled_indices = np.nonzero(row)
        top_board[filled_indices] = np.maximum(row_id + 1, top_board[filled_indices])
    return top_board


def simulate_drop(state, action):
    board, idx = state
    ori, col_idx = action

    # find board top
    top_board = get_top(board)
    col_start = col_idx
    block_width = pWidth[idx][ori]
    col_end = col_start + block_width
    top_board_at_block = top_board[col_start:col_end]

    # find block bottom
    bottom_block = np.array(pBottom[idx][ori])

    # calculate difference
    col_gap = top_board_at_block - bottom_block
    gap = max(col_gap)

    # place the block
    board_sim = np.copy(board)
    for col in range(col_start, col_end):
        j = col - col_start
        for i in range(pBottom[idx][ori][j], pTop[idx][ori][j]):
            row = int(i + gap)
            board_sim[row][col] = 1

    return board_sim


class Policy:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        self.q_func = QFunc(args)
        self.optimizer = optim.RMSprop(self.q_func.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        return

    def take_action(self, state, is_valid=True):

        print("\ttake_action:")

        action_space = get_action_space(state[1])

        # epsilon greedy
        if not is_valid and (random.random() < self.args.epsilon_g):
            print("taking random action")
            action = random.choice(action_space)
            print(action)
            return action

        # choose action with max Q
        max_q = -math.inf
        max_action = None

        qs = []

        print("\t\tqs:\t", end="")
        for action in action_space:
            this_q = self.get_Q(state, action).detach()
            print(this_q.data.numpy(), end=",")
            qs.append(this_q)
            if this_q > max_q:
                max_q = this_q
                max_action = action

        return max_action

    def get_Q(self, state, action):
        state_sim = simulate_drop(state, action)
        state_sim = np.expand_dims(state_sim, axis=0).astype(float)
        state_sim_var = Variable(torch.from_numpy(state_sim)).unsqueeze(0).float()
        Q = self.q_func(state_sim_var)
        return Q

    def learn(self, samples):
        # compute ys
        print("\tin learning:")
        states, actions, next_states, rewards, is_ends = samples
        ys = []
        for i in range(self.args.batch_size):
            if is_ends[i]:
                ys.append(rewards[i])
            else:
                action_space = get_action_space(next_states[i][1])
                max_q = -math.inf
                for next_action in action_space:
                    max_q = max(max_q, self.get_Q(next_states[i], next_action).detach())
                ys.append(rewards[i] + self.args.gamma * max_q)

        # compute Q
        loss = 0
        print("\t\tys:\t", end="")
        for i in range(self.args.batch_size):
            Q = self.get_Q(states[i], actions[i])
            print("(Q:", Q.data.numpy(), end=",")
            print("ys:", ys[i], end=") ")
            loss += (ys[i] - Q) ** 2
        loss /= self.args.batch_size

        self.logger.add_loss(loss.data.numpy().item())
        if len(self.logger.loss_list) % 10 == 0:
            self.logger.plot_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_params(self):
        torch.save(self.q_func.state_dict(), os.path.join(self.args.save_dir, "mymodel"))


class Logger:
    def __init__(self, args):
        self.args = args
        self.loss_list = []
        self.reward_list = []
        self.loss_figure = plt.figure("loss")
        self.reward_figure = plt.figure("reward")

    def add_loss(self, loss):
        self.loss_list.append(loss)

    def add_reward(self, reward):
        self.reward_list.append(reward)

    def plot_loss(self):
        plt.figure('loss')
        plt.plot(self.loss_list)
        plt.savefig(os.path.join(self.args.save_dir, 'loss.png'))

    def plot_reward(self):
        plt.figure('reward')
        plt.plot(self.reward_list)
        plt.savefig(os.path.join(self.args.save_dir, 'reward.png'))


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.state_list = []
        self.action_list = []
        self.cleared_list = []
        self.is_end_list = []
        self.reward_list = []
        self.count = 0
        self.valid_idx_list = []
        self.start_index = 0
        self.prev_end = -1

    def rel_index(self, index):
        return index - self.start_index

    def get_size(self):
        return len(self.valid_idx_list)

    def add(self, state, action, rows_cleared, is_end):
        self.state_list.append(state)
        self.action_list.append(action)
        self.cleared_list.append(rows_cleared)
        self.is_end_list.append(is_end)

        # calculate reward
        if self.count == self.prev_end + 1:
            self.reward_list.append(0)
        else:
            self.reward_list.append(self.calc_reward(self.rel_index(self.count)))

        if not is_end:
            self.valid_idx_list.append(self.count)

        # cacluate accumulative reward
        if is_end:
            reward_sum = sum(self.reward_list[self.rel_index(self.prev_end) + 1: self.rel_index(self.count)])
            self.args.logger.add_reward(reward_sum)
            self.prev_end = self.count

        # keep replay buffer udner capacity
        if len(self.state_list) > self.args.max_capacity:
            self.state_list = self.state_list[100:]
            self.action_list = self.action_list[100:]
            self.cleared_list = self.cleared_list[100:]
            self.is_end_list = self.is_end_list[100:]
            self.start_index += 100
            idx = np.argmax(self.valid_idx_list >= self.start_index)
            self.valid_idx_list = self.valid_idx_list[idx:]

        self.count += 1

    def sample(self, num_samples):
        indices = np.random.choice(self.valid_idx_list[:-1], num_samples, replace=False)
        indices -= self.start_index
        sampled_states = [self.state_list[i] for i in indices]
        sampled_actions = [self.action_list[i] for i in indices]
        sampled_next_state = [self.state_list[i + 1] for i in indices]
        sampled_rewards = [self.reward_list[i + 1] for i in indices]
        sampled_is_end = [self.is_end_list[i + 1] for i in indices]
        return [sampled_states, sampled_actions, sampled_next_state, sampled_rewards, sampled_is_end]

    def visualize_replaybuffer(self):
        print("end_list", self.is_end_list)
        print("valid_idx_list", self.valid_idx_list)
        sampled_states, sampled_actions, sampled_next_state, sampled_rewards, sampled_is_end = self.sample(10)
        print("sampled_is_end", sampled_is_end)
        print("sampled_rewards", sampled_rewards)

    def calc_reward(self, i):
        if self.is_end_list[i]:
            reward = -10
        else:
            reward = 1 + (self.cleared_list[i] - self.cleared_list[i - 1]) * 5
        return reward



def calculate_reward(state, next_state, is_end):
    reward = 0
    return reward


def collect_data(args, policy, replay_buffer, num_games=1):
    gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
    listener = PythonListener(args, gateway, policy, replay_buffer)
    gateway.entry_point.registerBenchListener(listener)
    gateway.entry_point.startGames(1, num_games)


def main():
    args = parse_args()
    args.logger = Logger(args)
    policy = Policy(args)
    replay_buffer = ReplayBuffer(args)
    collect_data(args, policy, replay_buffer, num_games=args.num_games)


def run_heuristic():
    args = parse_args()
    policy = HeuristicPolicy(args)
    replay_buffer = ReplayBuffer(args)
    collect_data(args, policy, replay_buffer, num_games=args.num_games)


def test_policy():
    args = parse_args()
    logger = Logger(args)
    policy = Policy(args)
    replay_buffer = ReplayBuffer(args)

    for i in range(2):

        # create some sample state, next piece
        next_piece = random.randint(0, 6)
        field = np.zeros((rows + 4, cols))
        field = field > 0
        is_end = False
        rows_cleared = 0
        state = [field, next_piece]

        # select action
        action = policy.take_action(state)

        # store transition
        replay_buffer.add(state, action, rows_cleared, is_end)

        # sample and update
        if len(replay_buffer.valid_idx_list) > args.batch_size:
            samples = replay_buffer.sample(args.batch_size)
            policy.learn(samples)


if __name__ == "__main__":
    # run_heuristic()
    main()
    # test_policy()
