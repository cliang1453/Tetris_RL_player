import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import random
import importlib
from py4j.java_gateway import JavaGateway, CallbackServerParameters
from params import *
from environment import *
import math
from model import QFunc
import torch
import torch.optim as optim
from torch.autograd import Variable
from termcolor import colored
from itertools import count


def parse_args():
    parser = argparse.ArgumentParser("RL for Tetris")
    parser.add_argument("--alg", type=str, default="", help="reinforcement learning algorithm")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.95, help="alpha")
    parser.add_argument("--eps", type=float, default=0.01, help="eps")
    parser.add_argument("--epsilon-g", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-games", type=int, default=10, help="number of games")
    parser.add_argument("--batch-size", type=int, default=32, help="number of games")
    parser.add_argument("--save-dir", type=str, default="log", help="directory to save policy and plots")
    parser.add_argument("--save-interval", type=int, default=10, help="interval to save validation plots")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--max-capacity", type=int, default=50000, help="maximum capacity of replay buffer")  # TODO: later change to 100000
    parser.add_argument("--learning-start", type=int, default=100, help="learning start after number of episodes")
    parser.add_argument("--num_collect_iter", type=int, default=100, help="number of iteration to collect data per one learning step")
    parser.add_argument("--num_target_update_iter", type=int, default=1, help="number of iterations to update target Q")
    # parser.add_argument("--use-cuda", type=bool, default=False, help="use cuda for training")
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
        self.validation_replay_buffer = ReplayBuffer(args, is_valid=True)
        self.game_count = 0

    def is_valid(self):
        if self.game_count % self.args.save_interval == 0:
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
        else:
            print("\t adding to validation replay buffer...")
            self.validation_replay_buffer.add(state, action, rows_cleared, is_end)

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


class Policy:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.q_func = QFunc(args).type(dtype)
        self.target_q_func = QFunc(args).type(dtype)
        self.optimizer = optim.RMSprop(self.q_func.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)

    def to_variable(self, array_in):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(array_in)).float().cuda()
        else:
            return Variable(torch.from_numpy(array_in)).float()

    def get_data(self, var_in):
        if torch.cuda.is_available():
            return var_in.data.cpu().numpy()
        else:
            return var_in.data.numpy()

    def take_action(self, state, strategy="epsilon_greedy"):
        action_space = get_action_space(state[1])

        # epsilon greedy
        if (strategy == "epsilon_greedy" and (random.random() < self.args.epsilon_g)) or strategy == 'random':
            action = random.choice(action_space)
            return action

        qs = self.get_Qs(state, action_space, q_func=self.q_func).detach()
        qs = self.get_data(qs)
        max_action_idx = np.argmax(qs)
        max_action = action_space[max_action_idx]

        return max_action

    def get_Qs(self, states, actions, q_func, diff_states=False):
        if not diff_states:
            states = [states] * len(actions)
        state_sim_list = []
        for state, action in zip(states, actions):
            state_sim = simulate_drop(state, action)
            state_sim = np.expand_dims(state_sim, axis=0).astype(float)
            state_sim_list.append(state_sim)
        states_sim = np.stack(state_sim_list, axis=0)
        states_sim_var = self.to_variable(states_sim)
        Qs = q_func(states_sim_var)
        return Qs

    def learn(self, samples):

        # compute ys
        states, actions, next_states, rewards, is_ends = samples
        ys = []
        for i in range(self.args.batch_size):
            if is_ends[i]:
                ys.append(rewards[i])
            else:
                action_space = get_action_space(next_states[i][1])
                qs = self.get_Qs(next_states[i], action_space, q_func=self.target_q_func).detach()
                max_q = np.asscalar(np.amax(self.get_data(qs)))
                ys.append(rewards[i] + self.args.gamma * max_q)

        # compute Q
        loss = 0
        Qs = self.get_Qs(states, actions, diff_states=True, q_func=self.q_func)
        ys_var = self.to_variable(np.array(ys)).view(-1, 1)
        loss = torch.sum((Qs - ys_var) ** 2) / self.args.batch_size

        ave_q = np.average(self.get_data(Qs))
        self.args.logger.add_q(ave_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.get_data(loss).item()

    def update_target_q(self):
        self.target_q_func.load_state_dict(self.q_func.state_dict())

    def save_params(self):
        torch.save(self.q_func.state_dict(), os.path.join(self.args.save_dir, "mymodel.pth"))

    def load_params(self):
        self.q_func.load_state_dict(torch.load(os.path.join(self.args.save_dir, "mymodel.pth")))


class Logger:
    def __init__(self, args):
        self.args = args
        self.loss_list = []
        self.reward_list = []
        self.reward_validation_list = []
        self.q_list = []

    def add_loss(self, loss):
        self.loss_list.append(loss)

    def add_reward(self, reward, is_valid=False):
        if is_valid:
            self.reward_validation_list.append(reward)
        else:
            self.reward_list.append(reward)

    def add_q(self, q):
        self.q_list.append(q)

    def plot_loss(self):
        plt.figure('loss')
        plt.plot(self.loss_list)
        plt.savefig(os.path.join(self.args.save_dir, 'loss.png'))

    def plot_reward(self):
        plt.figure('reward')
        plt.plot(self.reward_list)
        plt.savefig(os.path.join(self.args.save_dir, 'reward.png'))

        plt.figure('validation reward')
        plt.plot(self.reward_validation_list)
        plt.savefig(os.path.join(self.args.save_dir, 'validation_reward.png'))

    def plot_q(self):
        plt.figure('q')
        plt.plot(self.q_list)
        plt.savefig(os.path.join(self.args.save_dir, 'q.png'))


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.state_list = [None] * self.args.max_capacity
        self.action_list = [None] * self.args.max_capacity
        self.is_end_list = [None] * self.args.max_capacity
        self.reward_list = [None] * self.args.max_capacity
        self.count = 0

    def rel_index(self, index):
        return index % self.args.max_capacity

    def add(self, state, action, reward, is_end):
        index = self.rel_index(self.count)
        self.state_list[index] = state
        self.action_list[index] = action
        self.reward_list[index] = reward
        self.is_end_list[index] = is_end
        self.count += 1

    def get_size(self):
        return min(self.count, self.args.max_capacity)

    def sample(self, num_samples):
        indices = np.random.choice(self.get_size(), num_samples, replace=False)
        sampled_states = [self.state_list[i] for i in indices]
        sampled_actions = [self.action_list[i] for i in indices]
        sampled_next_state = [self.state_list[self.rel_index(i + 1)] if not self.is_end_list[i] else None for i in indices]
        sampled_rewards = [self.reward_list[i] for i in indices]
        sampled_is_end = [self.is_end_list[i] for i in indices]
        return [sampled_states, sampled_actions, sampled_next_state, sampled_rewards, sampled_is_end]

    def visualize_replaybuffer(self):
        print("end_list", self.is_end_list)
        sampled_states, sampled_actions, sampled_next_state, sampled_rewards, sampled_is_end = self.sample(10)
        print("sampled_is_end", sampled_is_end)
        print("sampled_rewards", sampled_rewards)


def collect_data(args, policy, replay_buffer, num_games=1):
    gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
    listener = PythonListener(args, gateway, policy, replay_buffer)
    gateway.entry_point.registerBenchListener(listener)
    gateway.entry_point.startGames(1, num_games)


def main():
    # args = parse_args()
    # args.logger = Logger(args)
    # policy = Policy(args)
    # replay_buffer = ReplayBuffer(args)
    # collect_data(args, policy, replay_buffer, num_games=args.num_games)
    train()


def run_heuristic():
    args = parse_args()
    policy = HeuristicPolicy(args)
    replay_buffer = ReplayBuffer(args)
    collect_data(args, policy, replay_buffer, num_games=args.num_games)


def test_policy():
    args = parse_args()
    args.logger = Logger(args)
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


def test_replay_buffer():
    args = parse_args()
    args.logger = Logger(args)
    replay_buffer = ReplayBuffer(args)

    for game in range(100):
        for t in range(50):
            field = np.zeros((rows, cols))
            state = [field, 0]
            action = [0, 0]
            rows_cleared = 0
            is_end = True if t == 49 else False
            replay_buffer.add(state, action, rows_cleared, is_end)


def train():
    args = parse_args()
    args.logger = Logger(args)
    policy = Policy(args)
    replay_buffer = ReplayBuffer(args)
    env = TetrisGame(args)

    for episode in range(args.num_episodes):

        if episode < args.learning_start:
            strategy = "random"
        else:
            if episode % args.save_interval == 0:
                strategy = "validation"
            else:
                strategy = "epsilon_greedy"

        if strategy == "validation":
            policy.save_params()

        # collect data
        reward_accum_list = []
        for game in range(args.num_games):
            print(colored("=" * 40 + "Episode" + str(episode) + " Game " + str(game) + " " + strategy + "=" * 40, 'red'))
            env.reset()
            reward_accum = 0
            for t in count():
                state = [env.field, env.next_piece]
                rows_cleared_prev = env.rows_cleared
                action = policy.take_action(state, strategy)
                next_piece, field, rows_cleared, is_end = env.step(action)
                reward = calc_reward(rows_cleared_prev, rows_cleared, is_end)
                reward_accum += reward
                if strategy != "validation":
                    # store transition
                    replay_buffer.add(state, action, reward, is_end)

                if is_end:
                    reward_accum_list.append(reward_accum)
                    break
        if strategy != "random":
            args.logger.add_reward(np.average(reward_accum_list), is_valid=(strategy == "validation"))

        # learn
        if (episode >= args.learning_start) and (replay_buffer.get_size() > args.batch_size) and (strategy == "epsilon_greedy"):
            print("learning...")
            loss_list = []
            for i in range(args.num_games):
                samples = replay_buffer.sample(args.batch_size)
                loss = policy.learn(samples)
                loss_list.append(loss)
            args.logger.add_loss(np.average(loss_list))

        # make plot
        if strategy != "random":
            if len(args.logger.loss_list) % 10 == 0:
                args.logger.plot_loss()
                args.logger.plot_reward()
                args.logger.plot_q()

        # update target q
        if (strategy != "random") and (episode % args.num_target_update_iter == 0):
            print("updating target q")
            policy.update_target_q()


if __name__ == "__main__":
    # run_heuristic()
    main()
    # test_policy()
    # test_replay_buffer()
