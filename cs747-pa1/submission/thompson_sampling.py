import numpy as np

def thompson_sampling(arms, randomSeed, horizon, epsilon):
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

    for t in range(horizon):
        arm = 0
        max_sampled = -1
        for i in range(n):
            s = values[i]*count[i]
            f = count[i]-s
            sampled = np.random.beta(s+1, f+1)
            if(sampled>max_sampled):
                arm = i
                max_sampled = sampled

        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    result = "thompson-sampling, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(horizon) + ", " + str(horizon*max(arms)-REW)
    return result