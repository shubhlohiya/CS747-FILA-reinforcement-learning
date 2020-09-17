import numpy as np

def ucb(arms, randomSeed, horizon, epsilon=0):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: Dummy Parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(n):
        r = np.random.binomial(1, arms[t])
        count[t] += 1
        values[t] += r
        REW+=r

    ucbs = np.empty(n)
    for t in range(n, horizon):
        for i in range(n):
            ucbs[i] = values[i] + np.sqrt(2 * np.log(t)/count[i])

        arm = np.random.choice(np.where(ucbs==ucbs.max())[0])
        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    result = "ucb, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(horizon) + ", " + str(horizon*max(arms)-REW)
    return result