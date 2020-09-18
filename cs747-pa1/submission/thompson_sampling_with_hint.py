import numpy as np

def thompson_sampling_with_hint(arms, randomSeed, horizon, epsilon):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: Dummy Parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    bestval = np.max(arms)
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(horizon):
        arm = 0
        max_sampled = -1
        for i in range(n):
            s1 = values[i]*count[i]
            f1 = count[i]-s1
            s2 = count[i]*values[i]+1
            f2 = count[i]*bestval+1
            sampled = np.random.beta(s1+1, f1+1)*np.random.beta(s2, f2)
            if(sampled>max_sampled):
                arm = i
                max_sampled = sampled

        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    result = "thompson-sampling-with-hint, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(horizon) + ", " + str(horizon*max(arms)-REW)
    return result