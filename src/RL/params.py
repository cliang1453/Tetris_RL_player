cols = 10
rows = 21
num_pieces = 7

pWidth = [
    [2],
    [1, 4],
    [2, 3, 2, 3],
    [2, 3, 2, 3],
    [2, 3, 2, 3],
    [3, 2],
    [3, 2]
]

pHeight = [
    [2],
    [4, 1],
    [3, 2, 3, 2],
    [3, 2, 3, 2],
    [3, 2, 3, 2],
    [2, 3],
    [2, 3]
]
pBottom = [
    [[0, 0]],
    [[0], [0, 0, 0, 0]],
    [[0, 0], [0, 1, 1], [2, 0], [0, 0, 0]],
    [[0, 0], [0, 0, 0], [0, 2], [1, 1, 0]],
    [[0, 1], [1, 0, 1], [1, 0], [0, 0, 0]],
    [[0, 0, 1], [1, 0]],
    [[1, 0, 0], [0, 1]]
]
pTop = [
    [[2, 2]],
    [[4], [1, 1, 1, 1]],
    [[3, 1], [2, 2, 2], [3, 3], [1, 1, 2]],
    [[1, 3], [2, 1, 1], [3, 3], [2, 2, 2]],
    [[3, 2], [2, 2, 2], [2, 3], [1, 2, 1]],
    [[1, 2, 2], [3, 2]],
    [[2, 2, 1], [2, 3]]
]


def get_action_space(idx):
    num_ori = len(pWidth[idx])
    res = []
    for ori in range(num_ori):
        num_valid_cols = cols - pWidth[idx][ori] + 1
        for col_idx in range(num_valid_cols):
            res.append((ori, col_idx))
    return res
