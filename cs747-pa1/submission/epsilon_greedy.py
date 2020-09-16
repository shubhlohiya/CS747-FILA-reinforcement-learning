import numpy as np

def epsilon_greedy(arms, randomSeed, horizon, epsilon):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seem to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: The exploration parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = 0
    values = np.array([0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(n):
        if t==horizon:
            break
        count[t] += 1
        values[t] = np.random.binomial(p = arms[t])

    T = horizon-np.sum(count)
    for t in range(T):
        explore = np.random.binomial(p=epsilon)==1
        if explore:
            arm = np.random.randint(0, n)
            count[arm] += 1
            values[arm] = values[arm] + (np.random.binomial(p=arms[arm])-values[arm])/count[arm]

        else:
            np.argwhere







