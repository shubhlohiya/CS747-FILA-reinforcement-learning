import sys
from epsilon_greedy import epsilon_greedy
from ucb import ucb
from kl_ucb import kl_ucb
from thompson_sampling import thompson_sampling
from thompson_sampling_with_hint import thompson_sampling_with_hint

n = len(sys.argv)

instance_path, algorithm, randomSeed, epsilon, horizon = "../instances/i-2.txt", "kl-ucb", 0, 0.333, 200

for i in range(1,n):
    if sys.argv[i]== "--instance":
        i+=1
        instance_path = sys.argv[i]
    elif sys.argv[i] == "--algorithm":
        i+=1
        algorithm = sys.argv[i]

    elif sys.argv[i] == "--randomSeed":
        i+=1
        randomSeed = int(sys.argv[i])

    elif sys.argv[i] == "--epsilon":
        i+=1
        epsilon = float(sys.argv[i])

    elif sys.argv[i] == "--horizon":
        i+=1
        horizon = int(sys.argv[i])

with open(instance_path) as f:
    arms = [line.strip() for line in f]
    arms = list(map(float, arms))

result = instance_path + ", "

if algorithm == "epsilon-greedy":
    result += epsilon_greedy(arms, randomSeed, horizon, epsilon) + "\n"
elif algorithm == "ucb":
    result += ucb(arms, randomSeed, horizon, epsilon) + "\n"
elif algorithm == "kl-ucb":
    result += kl_ucb(arms, randomSeed, horizon, epsilon) + "\n"
elif algorithm == "thompson-sampling":
    result += thompson_sampling(arms, randomSeed, horizon, epsilon) + "\n"
elif algorithm == "thompson-sampling-with-hint":
    result += thompson_sampling_with_hint(arms, randomSeed, horizon, epsilon, hint=max(arms)) + "\n"

print(result)