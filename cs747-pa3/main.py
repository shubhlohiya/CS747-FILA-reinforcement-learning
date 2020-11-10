import argparse, numpy as np
import matplotlib.pyplot as plt
from agent import Agent
parser = argparse.ArgumentParser()

def run(algorithm, kings_moves = False, stochastic=False):
    data = []
    for i in range(10):
        np.random.seed(i)
        data.append(Agent(algorithm=algorithm, kings_moves=kings_moves, stochastic=stochastic).res)

    data = np.array(data)
    data = np.mean(data, axis=0)
    return (data, np.arange(1, len(data)+1))

if __name__ == "__main__":
    parser.add_argument("--task", "-t", type=int, default=5, help="Task number from {2,3,4,5}")
    parser.add_argument("--stochastic", "-s", type=int, default=None, help="Include stochastic wind or not. Enter 0 or 1")
    parser.add_argument("--kingsmoves", "-k", type=int, default=None, help="Allow King's moves or not. Enter 0 or 1")
    parser.add_argument("--algorithm", "-a", type=str, default=None, help="Choices are from {sarsa, expected-sarsa, q-learning, compare}.\n'compare' will compare all three algorithms on a plot")
    args = parser.parse_args()
    plt.rcParams['figure.figsize'] = 10, 10

    if  args.algorithm and (args.stochastic!=None) and (args.kingsmoves!=None):
        if args.algorithm == "compare":
            plt.plot(*run("sarsa", kings_moves=args.kingsmoves, stochastic=args.stochastic), label="Sarsa")
            plt.plot(*run("q-learning", kings_moves=args.kingsmoves, stochastic=args.stochastic), label="Q-Learning")
            plt.plot(*run("expected-sarsa", kings_moves=args.kingsmoves, stochastic=args.stochastic), label="Expected Sarsa")
            plt.legend()
        else:
            plt.plot(*run(args.algorithm, kings_moves=args.kingsmoves, stochastic=args.stochastic))
        plt.title(f"Windy Gridworld Agent: king's moves = {bool(args.kingsmoves)}, stochastic wind = {bool(args.stochastic)}, algorithm = {args.algorithm}\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.grid()
        plt.show()

    elif args.task == 2:
        plt.plot(*run("sarsa"))
        plt.title("Task 2: Windy Gridworld Agent with 4 moves - Sarsa(0)\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.grid()
        plt.savefig("plots/task2.png")
        plt.show()

    elif args.task == 3:
        plt.plot(*run("sarsa", kings_moves=True))
        plt.title("Task 3: Windy Gridworld Agent with King's moves - Sarsa(0)\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.grid()
        plt.savefig("plots/task3.png")
        plt.show()

    elif args.task == 4:
        plt.plot(*run("sarsa", kings_moves=True, stochastic=True))
        plt.title("Task 4: Windy Gridworld Agent with King's moves and stochastic wind - Sarsa(0)\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.grid()
        plt.savefig("plots/task4.png")
        plt.show()

    elif args.task == 5:
        plt.plot(*run("sarsa"), label="Sarsa")
        plt.plot(*run("q-learning"), label="Q-Learning")
        plt.plot(*run("expected-sarsa"), label="Expected Sarsa")
        plt.title("Task 5: Windy Gridworld Agent with 4 moves - Algorithm Comparision\n")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.legend()
        plt.grid()
        plt.savefig("plots/task5.png")
        plt.show()

    else: 
        print("Incorrect command line arguments, please try again!")