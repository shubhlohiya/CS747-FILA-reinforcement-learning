import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputDataT2.txt", header = None, delimiter=", ")
df.columns = ['instance', 'algorithm', 'seed', 'epsilon', 'horizon', 'regret']

data = df.groupby(['instance', 'algorithm', 'horizon']).mean().reset_index()


algos = ['thompson-sampling', 'thompson-sampling-with-hint']
instances = [f"../instances/i-{k}.txt" for k in range(1,4)]

plt.rcParams['figure.figsize'] = 12, 8

for instance in instances:
    plt.xscale("log")
    for algo in algos:
        temp=data[(data.instance==instance) & (data.algorithm==algo)]
        x = list(temp.horizon)
        y = list(temp.regret)
        plt.plot(x,y, label=algo)        
    plt.title("Instance " + str(instance[-5])+"\n", fontsize=20)
    plt.legend()
    plt.savefig("plots/T2_instance_" + str(instance[-5])+".png")
    plt.show()