import numpy as np
from params import *
import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_top(board):
    top_board = np.zeros(cols)
    for row_id in range(rows):
        row = board[row_id]
        filled_indices = np.nonzero(row)
        top_board[filled_indices] = np.maximum(row_id + 1, top_board[filled_indices])
    return top_board


def simulate_drop(state, action, in_game=False):
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
    game_end = False
    board_sim = np.copy(board)
    for col in range(col_start, col_end):
        j = col - col_start
        for i in range(pBottom[idx][ori][j], pTop[idx][ori][j]):
            row = int(i + gap)
            board_sim[row][col] = 1
            if row >= rows:
                game_end = True

    if in_game:
        # remove filled rows
        cleared = 0
        if not game_end:
            filled_rows = []
            for i in range(rows):
                if np.all(board_sim[i]):
                    filled_rows.append(i)

            cleared = len(filled_rows)
            board_sim = np.delete(board_sim, filled_rows, axis=0)
            board_sim = np.append(board_sim, np.zeros((cleared, cols)), axis=0)

        return board_sim, game_end, cleared
    else:
        return board_sim


class TetrisGame:
    def __init__(self, do_visualize=False):
        self.field = np.zeros((rows + 4, cols))
        self.next_piece = None
        self.generate_random_piece()
        self.rows_cleared = 0
        self.is_end = False
        self.do_visualize = do_visualize
        if do_visualize:
            self.fig, self.ax = plt.subplots(1)

    def reset(self):
        self.field = np.zeros((rows + 4, cols))
        self.rows_cleared = 0
        self.is_end = False
        self.generate_random_piece()
        if self.do_visualize:
            self.visualize()
        return self.next_piece, self.field, self.rows_cleared, self.is_end

    def generate_random_piece(self):
        self.next_piece = random.randint(0, num_pieces - 1)

    def step(self, action):
        # drop the block
        field_new, game_end, cleared = simulate_drop([self.field, self.next_piece], action, in_game=True)
        self.rows_cleared += cleared
        self.is_end = game_end
        self.field = field_new
        self.generate_random_piece()
        if self.do_visualize:
            self.visualize()
        return self.next_piece, self.field, self.rows_cleared, self.is_end

    def visualize(self):
        # plt.clf()
        # self.fig, self.ax = plt.subplots(1)
        # self.fig.canvas.flush_events()
        self.ax.imshow(self.field[::-1])
        plt.pause(0.05)


if __name__ == "__main__":
    env = TetrisGame(do_visualize=False)
    next_piece, field, rows_cleared, is_end = env.reset()
    for t in count():
        env.next_piece = 0
        print("=" * 40, t, "=" * 40)
        # print('next_piece', next_piece)
        # print('field')
        # print(field[::-1])
        print('rows_cleared', rows_cleared)
        print('is_end', is_end)
        if is_end:
            break
        action_space = get_action_space(next_piece)
        # action = random.choice(action_space)
        action = [0, t % 5 * 2]

        next_piece, field, rows_cleared, is_end = env.step(action)
        print('action', action)
