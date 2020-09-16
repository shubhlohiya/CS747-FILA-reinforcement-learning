import sys
from epsilon_greedy import epsilon_greedy
from ucb import ucb
from kl_ucb import kl_ucb
from thompson_sampling import thompson_sampling
from thompson_sampling_with_hint import thompson_sampling_with_hint

n = len(sys.argv)

instance_path, algorithm, randomSeed, epsilon, horizon = "", "", 0, 0, 200

for i in range(1,n):
    if sys.argv[i]== "--instance":
        i+=1
        instance_path = sys.argv[i]
    elif sys.argv[i] == "--algorithm":
        i+=1
        algorithm = sys.argv[i]

    elif sys.argv[i] == "--randomSeed":
        i+=1
        randomSeed = sys.argv[i]

    elif sys.argv[i] == "--epsilon":
        i+=1
        epsilon = sys.argv[i]

    elif sys.argv[i] == "--horizon":
        i+=1
        horizon = sys.argv[i]

with open(instance_path) as f:
    arms = [line.strip() for line in f]
    arms = list(map(float, arm))

if algorithm == "epsilon-greedy":
    epsilon_greedy(arms, randomSeed, horizon, epsilon)
elif algorithm == "ucb":
    ucb(arms, randomSeed, horizon, epsilon)
elif algorithm == "kl-ucb":
    kl_ucb(arms, randomSeed, horizon, epsilon)
elif algorithm == "thompson-sampling":
    thompson_sampling(arms, randomSeed, horizon, epsilon)
elif algorithm == "thompson-sampling-with-hint":
    thompson_sampling_with_hint(arms, randomSeed, horizon, epsilon)
