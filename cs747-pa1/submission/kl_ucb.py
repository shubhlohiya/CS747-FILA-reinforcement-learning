import numpy as np

def KL(p, q):
    if p==q:
        return 0
    if q==0 or q==1:
        return float("inf")
    if p==1:
        return np.log(1/q)
    if p==0:
        return -np.log(1-q)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def get_kl_ucb(value, count, t, c=3):
    delta = 0.005
    target = (np.log(t) + c*np.log(np.log(t)))/count
    low, high = value, 1
    mid = (low+high)/2
    res = KL(value, mid)
    while abs(res-target)>delta and mid-low>delta:
        if res>target:
            high = mid
        else:
            low = mid
        mid = (low + high) / 2
        res = KL(value, mid)
    return round(mid, 3)

def kl_ucb(arms, randomSeed, horizon, epsilon=0):
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
        if t==horizon:
            break
        r = np.random.binomial(1, arms[t])
        count[t] += 1
        values[t] += r
        REW+=r

    done = np.sum(count)
    kl_ucbs = np.empty(n)
    for t in range(done, horizon):
        for i in range(n):
            kl_ucbs[i] = get_kl_ucb(values[i], count[i], t)

        arm = np.random.choice(np.where(kl_ucbs==kl_ucbs.max())[0])
        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    result = "kl-ucb, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(horizon) + ", " + str(horizon*max(arms)-REW)
    return result