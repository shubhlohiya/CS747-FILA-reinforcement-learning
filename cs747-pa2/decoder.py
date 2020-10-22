import argparse, numpy as np
parser = argparse.ArgumentParser()

def decode(grid_path, policy_path):
    # collect mdp data from maze
    maze = open(grid_path).read().strip().split("\n")
    maze = np.array(list(map(lambda x: x.split(), maze)), dtype=int)
    ns = -1
    na = 4  # 0-N, 1-S, 2-E, 3-W
    a, b = maze.shape  # axb maze
    state_names = -np.ones(maze.shape, dtype=int)
    end = []
    for i in range(1, a - 1):
        for j in range(1, b - 1):
            val = maze[i, j]
            if val == 0:
                ns += 1
                state_names[i, j] = ns
            elif val == 2:
                ns += 1
                start = ns
                state_names[i, j] = ns
            elif val == 3:
                ns += 1
                end.append(ns)
                state_names[i, j] = ns
    ns += 1

    optimal_data = open(policy_path).read().strip().split("\n")
    pi = [eval(line.split()[-1]) for line in optimal_data]
    curr = start
    path = ""
    while curr not in end:
        path += get_action(pi[curr]) + " "
        curr = next_state(curr, pi[curr], state_names)
    print(path.strip())

def get_action(a):
    if a==0:
        return 'N'
    if a==1:
        return 'S'
    if a==2:
        return 'E'
    return 'W'
def next_state(s, a, state_names):
    i, j = np.squeeze(np.where(state_names==s))
    if a == 0:
        return state_names[i-1, j]
    if a == 1:
        return state_names[i+1, j]
    if a == 2:
        return state_names[i, j+1]
    return state_names[i, j-1]

if __name__ == "__main__":
    parser.add_argument("--grid", type=str, default="./data/maze/grid10.txt")
    parser.add_argument("--value_policy", type=str, default="./value_and_policy_file")
    args = parser.parse_args()
    decode(args.grid, args.value_policy)