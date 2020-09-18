import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputDataT1.txt", header = None, delimiter=", ")
df.columns = ['instance', 'algorithm', 'seed', 'epsilon', 'horizon', 'regret']

data = df.groupby(['instance', 'algorithm', 'horizon']).mean().reset_index()


algos = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']
instances = [f"../instances/i-{k}.txt" for k in range(1,4)]

plt.rcParams['figure.figsize'] = 12, 8

for instance in instances:
    plt.xscale("log")
    for algo in algos:
        temp=data[(data.instance==instance) & (data.algorithm==algo)]
        x = list(temp.horizon)
        y = list(temp.regret)
        plt.plot(x,y, label=algo)        
    plt.xlabel("Time Steps (Logarithmic Scale)", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title("Instance " + str(instance[-5])+"\n", fontsize=20)
    plt.legend()
    plt.savefig("plots/T1_instance_" + str(instance[-5])+".png")
    plt.show()