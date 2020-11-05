import argparse, numpy as np
import matplotlib.pyplot as plt
from agent import Solve
parser = argparse.ArgumentParser()

def run(algorithm, kings_moves = False, stochastic=False):
    np.random.seed(0)
    data = []
    for i in range(10):
        np.random.seed(i)
        data.append(Solve(algorithm=algorithm, kings_moves=kings_moves, stochastic=stochastic).res)

    data = np.array(data)
    data = np.mean(data, axis=0)
    return (data, np.arange(1, len(data)+1))

if __name__ == "__main__":
    parser.add_argument("--task", type=int, default=5)
    args = parser.parse_args()
    task = args.task
    plt.rcParams['figure.figsize'] = 10, 10

    if task == 2:
        plt.plot(*run("sarsa"))
        plt.title("Task 2: Windy Gridworld Agent with 4 moves - Sarsa(0)\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.savefig("plots/task2.png")
        plt.show()

    elif task == 3:
        plt.plot(*run("sarsa", kings_moves=True))
        plt.title("Task 3: Windy Gridworld Agent with King's moves - Sarsa(0)\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.savefig("plots/task3.png")
        plt.show()

    elif task == 4:
        plt.plot(*run("sarsa", kings_moves=True, stochastic=True))
        plt.title("Task 4: Windy Gridworld Agent with King's moves and stochastic wind - Sarsa(0)\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.savefig("plots/task4.png")
        plt.show()

    else:
        plt.plot(*run("sarsa"), label="Sarsa")
        plt.plot(*run("q-learning"), label="Q-Learning")
        plt.plot(*run("expected-sarsa"), label="Expected Sarsa")
        plt.title("Task 5: Windy Gridworld Agent with 4 moves - Algorithm Comparision\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.legend()
        plt.savefig("plots/task5.png")
        plt.show()