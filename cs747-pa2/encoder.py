import argparse, numpy as np
parser = argparse.ArgumentParser()

def encode(path):
    # collect mdp data from maze
    maze = open(path).read().strip().split("\n")
    maze = np.array(list(map(lambda x: x.split(), maze)), dtype=int)
    ns = -1
    na = 4  # 0-N, 1-S, 2-E, 3-W
    a, b = maze.shape  # axb maze
    state_names = -np.ones(maze.shape, dtype=int)
    mdp = ""
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
                end.append(str(ns))
                state_names[i, j] = ns
    ns += 1
    mdp += f"numStates {ns}\n"
    mdp += f"numActions {na}\n"
    mdp += f"start {start}\n"
    mdp += "end " + " ".join(end) + "\n"

    for i in range(1, a - 1):
        for j in range(1, b - 1):
            if maze[i, j] != 1:
                if maze[i, j] == 3:
                    continue
                mdp += get_transitions(i, j, maze, state_names)

    mdp += "mdptype episodic\n"
    mdp += "discount  0.9"
    print(mdp)

def get_transitions(i, j, maze, state_names):
    s = []
    s_values = []
    a_values = []
    if maze[i-1, j]!=1:
        s.append(state_names[i-1, j])
        s_values.append(maze[i-1, j])
        a_values.append(0)
    if maze[i+1, j]!=1:
        s.append(state_names[i+1, j])
        s_values.append(maze[i+1, j])
        a_values.append(1)
    if maze[i, j+1]!=1:
        s.append(state_names[i, j+1])
        s_values.append(maze[i, j+1])
        a_values.append(2)
    if maze[i, j-1]!=1:
        s.append(state_names[i, j-1])
        s_values.append(maze[i, j-1])
        a_values.append(3)
    transitions = ""
    for t in range(len(s)):
        transitions += f"transition {state_names[i,j]} {a_values[t]} {s[t]} -1.0 1.0\n"
    return transitions

if __name__ == "__main__":
    parser.add_argument("--grid", type=str, default="./data/maze/grid10.txt")
    args = parser.parse_args()
    encode(args.grid)