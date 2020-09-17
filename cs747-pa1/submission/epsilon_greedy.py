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
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    # for t in range(n):
    #     if t==horizon:
    #         break
    #     r = np.random.binomial(1, arms[t])
    #     count[t] += 1
    #     values[t] += r
    #     REW+=r

    T = horizon#-np.sum(count)
    for t in range(T):
        if np.random.random() < epsilon:
            arm = np.random.randint(0, n)
        else:
            arm = np.random.choice(np.where(values==values.max())[0])

        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    result = "epsilon-greedy, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(horizon) + ", " + str(horizon*max(arms)-REW)
    return result