import numpy as np

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def thompson_sampling_with_hint(arms, randomSeed, horizon, epsilon, hint):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: Dummy Parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    optimal_mean = hint
    var = optimal_mean*(1-optimal_mean)
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(n):
        r = np.random.binomial(1, arms[t])
        count[t] += 1
        values[t] += r
        REW+=r

    for t in range(n, horizon):
        arm = 0
        max_sampled = -1
        for i in range(n):
            sampled = gaussian(values[i], optimal_mean, var/np.sqrt(count[i]))
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