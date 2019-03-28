from py4j.java_gateway import JavaGateway, CallbackServerParameters
from train import *


class PythonListener(object):

    def __init__(self, args, gateway, policy):
        self.args = args
        self.gateway = gateway
        self.policy = policy
        self.game_count = 0

    def is_valid(self):
        if self.game_count % self.args.save_interval == 0:
            return True
        else:
            return False

    def notify(self, next_piece, field, rows_cleared, is_end):
        # print(colored("=" * 40 + str(self.game_count) + "=" * 40, 'red'))
        print(rows_cleared)
        # format state
        field = list(field)
        field = [list(row) for row in field]
        field = field + [[0] * cols for _ in range(4)]  # add 4 rows on top
        field = np.array(field)
        field = field > 0
        state = [field, next_piece]

        # select action
        if is_end:
            action = [0, 0]
        else:
            action = self.policy.take_action(state, strategy="validation")

        self.gateway.entry_point.takeAction(int(action[0]), int(action[1]))

    class Java:
        implements = ["org.py4j.smallbench.BenchListener"]


def collect_data(args, policy, num_games=1):
    gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
    listener = PythonListener(args, gateway, policy)
    gateway.entry_point.registerBenchListener(listener)
    gateway.entry_point.startGames(1, num_games)


if __name__ == "__main__":
    args = parse_args()
    args.logger = Logger(args)
    # policy = MagicPolicy(args)
    policy = Policy(args)
    policy.load_params()

    collect_data(args, policy)
