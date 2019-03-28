import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import random

from params import *
from environment import *
from model import *
import torch
import torch.optim as optim
from torch.autograd import Variable
from termcolor import colored
from itertools import count
import pickle


def parse_args():
    parser = argparse.ArgumentParser("RL for Tetris")
    parser.add_argument("--alg", type=str, default="DQN", help="reinforcement learning algorithm, DQN, Q_learning")
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
    parser.add_argument("--max-capacity", type=int, default=50000, help="maximum capacity of replay buffer")
    parser.add_argument("--learning-start", type=int, default=10, help="learning start after number of episodes")
    parser.add_argument("--num_target_update_iter", type=int, default=1, help="number of iterations to update target Q")
    parser.add_argument("--features", type=str, default="all", help="options are cnn, all, magic")
    parser.add_argument("--reward-type", type=str, default="all", help="options are all, cleared")
    parser.add_argument("--use-heuristic", action="store_true", help="using heuristic to collect data")
    args = parser.parse_args()

    return args


"""
exp1: --alg Q_learning --features cnn --reward-type all
exp2: --alg DQN --features cnn --reward-type all
exp3: --alg DQN --features cnn --reward-type cleared
exp4: --alg DQN --features magic --reward-type all --lr 0.001
exp5: --alg DQN --features all --reward-type all
exp6: --alg DQN --features cnn --reward-type all --use-heuristic
"""

"""
state: [ndarray(rows * cols), next_piecee]

"""


class MagicPolicy:
    def __init__(self, args):
        self.args = args
        self.magic_numbers = np.array([-0.510066, 0.760666, -0.35663, -0.184483])

    def take_action(self, state, strategy="validation"):
        action_space = get_action_space(state[1])
        if (strategy == "epsilon_greedy" and random.random() < self.args.epsilon_g) or strategy == 'random':
            action = random.choice(action_space)
            return action

        values = []
        features_list = []
        for action in action_space:
            board_sim, features = simulate_drop(state, action, get_feature=True)
            features = np.array(features)[:4]
            value = np.dot(self.magic_numbers, features)
            features_list.append(features)
            values.append(value)
        best_index = np.argmax(np.array(values))
        # print(values)
        # print(best_index)
        # print(features_list[best_index])
        return action_space[best_index]


class Policy:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        if self.args.features == "cnn":
            q_function = QFunc
        elif self.args.features == "magic":
            q_function = QFuncFeature
        else:
            q_function = QFuncAll

        self.q_func = q_function(args).type(dtype)
        self.target_q_func = q_function(args).type(dtype)
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
        if (strategy == "epsilon_greedy" and random.random() < self.args.epsilon_g) or strategy == 'random':
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
        feature_list = []

        get_feature = (self.args.features == "all") or (self.args.features == "magic")

        for state, action in zip(states, actions):
            state_sim, feature = simulate_drop(state, action, get_feature=get_feature)
            state_sim = np.expand_dims(state_sim, axis=0).astype(float)
            state_sim_list.append(state_sim)
            feature_list.append(feature)

        states_sim = np.stack(state_sim_list, axis=0)
        states_sim_var = self.to_variable(states_sim)

        if get_feature:
            features = np.stack(feature_list, axis=0)
            features_var = self.to_variable(features)

        if self.args.features == "cnn":
            Qs = q_func(states_sim_var)
        elif self.args.features == "all":
            Qs = q_func(states_sim_var, features_var)
        else:
            Qs = q_func(features_var)
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
                if self.args.alg == "DQN":
                    qs = self.get_Qs(next_states[i], action_space, q_func=self.target_q_func).detach()
                else:
                    qs = self.get_Qs(next_states[i], action_space, q_func=self.q_func).detach()
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
        if self.args.alg == "DQN":
            self.target_q_func.load_state_dict(self.q_func.state_dict())

    def save_params(self):
        torch.save(self.q_func.state_dict(), os.path.join(self.args.save_dir, "mymodel.pth"))

    def load_params(self):
        if torch.cuda.is_available():
            self.q_func.load_state_dict(torch.load(os.path.join(self.args.save_dir, "mymodel.pth"), map_location='gpu'))
        else:
            self.q_func.load_state_dict(torch.load(os.path.join(self.args.save_dir, "mymodel.pth"), map_location='cpu'))


class Logger:
    def __init__(self, args):
        self.args = args
        self.loss_list = []
        self.reward_list = []
        self.reward_validation_list = []
        self.q_list = []
        self.cleared_list = []

    def add_loss(self, loss):
        self.loss_list.append(loss)

    def add_reward(self, reward, is_valid=False):
        if is_valid:
            self.reward_validation_list.append(reward)
        else:
            self.reward_list.append(reward)

    def add_q(self, q):
        self.q_list.append(q)

    def add_cleared(self, cleared):
        self.cleared_list.append(cleared)

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

    def log_all(self):
        with open(os.path.join(self.args.save_dir, 'q.pkl'), 'wb') as f:
            pickle.dump(self.q_list, f)
        with open(os.path.join(self.args.save_dir, 'validation_reward.pkl'), 'wb') as f:
            pickle.dump(self.reward_validation_list, f)
        with open(os.path.join(self.args.save_dir, 'reward.pkl'), 'wb') as f:
            pickle.dump(self.reward_list, f)
        with open(os.path.join(self.args.save_dir, 'loss.pkl'), 'wb') as f:
            pickle.dump(self.loss_list, f)
        with open(os.path.join(self.args.save_dir, 'cleared.pkl'), 'wb') as f:
            pickle.dump(self.cleared_list, f)


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


def main():
    train()


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
    if args.use_heuristic:
        heuristic_policy = MagicPolicy(args)
    replay_buffer = ReplayBuffer(args)
    env = TetrisGame(args)
    max_cleared = 0

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
        rows_cleared_list = []
        for game in range(args.num_games):
            env.reset()
            reward_accum = 0
            for t in count():
                state = [env.field, env.next_piece]
                rows_cleared_prev = env.rows_cleared
                if (strategy != "validation") and args.use_heuristic and (game < 2):
                    action = heuristic_policy.take_action(state, strategy)
                else:
                    action = policy.take_action(state, strategy)
                next_piece, field, rows_cleared, is_end = env.step(action)
                reward = calc_reward(rows_cleared_prev, rows_cleared, is_end, reward_type=args.reward_type)
                reward_accum += reward
                if strategy != "validation":
                    # store transition
                    replay_buffer.add(state, action, reward, is_end)

                if is_end:
                    max_cleared = max(max_cleared, rows_cleared)
                    print(colored(
                        "=" * 20 + "Episode" + str(episode) + " Game " + str(game) + " " + strategy + "=" * 20 + str(rows_cleared) + "/" + str(max_cleared),
                        'red'))

                    reward_accum_list.append(reward_accum)
                    rows_cleared_list.append(rows_cleared)
                    break
        if strategy != "random":
            args.logger.add_reward(np.average(reward_accum_list), is_valid=(strategy == "validation"))
            args.logger.add_cleared(np.average(reward_accum_list))

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
            if len(args.logger.loss_list) % 100 == 0:
                args.logger.plot_loss()
                args.logger.plot_reward()
                args.logger.plot_q()
                args.logger.log_all()

        # update target q
        if (strategy != "random") and (episode % args.num_target_update_iter == 0):
            print("updating target q")
            policy.update_target_q()


def do_validation():
    args = parse_args()
    args.logger = Logger(args)
    policy = MagicPolicy(args)
    # policy = Policy(args)
    # policy.load_params()
    replay_buffer = ReplayBuffer(args)
    env = TetrisGame(args, do_visualize=False)

    env.reset()
    reward_accum = 0
    strategy = "validation"
    for t in count():
        state = [env.field, env.next_piece]
        rows_cleared_prev = env.rows_cleared
        action = policy.take_action(state, strategy)
        next_piece, field, rows_cleared, is_end = env.step(action)
        reward = calc_reward(rows_cleared_prev, rows_cleared, is_end)
        reward_accum += reward
        print(rows_cleared)
        if is_end:
            break


if __name__ == "__main__":
    # run_heuristic()
    main()
    # test_policy()
    # test_replay_buffer()
    # do_validation()
