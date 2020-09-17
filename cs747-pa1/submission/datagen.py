import numpy as np

def epsilon_greedy(arms, randomSeed, horizon, epsilon, pauses, instance, f):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: The exploration parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(horizon):
        if t in pauses:
            f.write("\n" + instance+", "+"epsilon-greedy, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(t) + ", " + str(t*max(arms)-REW))
        if np.random.random() < epsilon:
            arm = np.random.randint(0, n)
        else:
            arm = np.random.choice(np.where(values==values.max())[0])

        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    f.write("\n" + instance + ", " + "epsilon-greedy, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(horizon) + ", " + str(horizon * max(arms) - REW))
    pass

def ucb(arms, randomSeed, horizon, epsilon, pauses, instance, f):
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
        if t in pauses:
            f.write("\n" + instance+", "+"ucb, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(t) + ", " + str(t*max(arms)-REW))

        for i in range(n):
            ucbs[i] = values[i] + np.sqrt(2 * np.log(t)/count[i])

        arm = np.random.choice(np.where(ucbs==ucbs.max())[0])
        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    f.write("\n" + instance + ", " + "ucb, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(horizon) + ", " + str(horizon * max(arms) - REW))
    pass

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

    while high-low>=delta:
        mid = (low + high) / 2
        res = KL(value, mid)
        if target>res and target-res<=delta:
            return mid
        elif res > target:
            high = mid
        else:
            low = mid
    return low

def kl_ucb(arms, randomSeed, horizon, epsilon, pauses, instance, f):
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

    kl_ucbs = np.empty(n)
    for t in range(n, horizon):
        if t in pauses:
            f.write("\n" + instance+", "+"kl-ucb, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(t) + ", " + str(t*max(arms)-REW))
        for i in range(n):
            kl_ucbs[i] = get_kl_ucb(values[i], count[i], t)
        arm = np.argmax(kl_ucbs)
        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    f.write("\n" + instance + ", " + "kl-ucb, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(horizon) + ", " + str(horizon * max(arms) - REW))
    pass

def thompson_sampling(arms, randomSeed, horizon, epsilon, pauses, instance, file):
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
        if t in pauses:
            file.write("\n" + instance+", "+"thompson-sampling, " + str(randomSeed)+ ", " + str(epsilon) + ", "\
             + str(t) + ", " + str(t*max(arms)-REW))
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

    file.write("\n" + instance + ", " + "thompson-sampling, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(horizon) + ", " + str(horizon * max(arms) - REW))
    pass

instances = [f"../instances/i-{k}.txt" for k in range(1,4)]
seeds = range(50)
algorithms = [epsilon_greedy, ucb, kl_ucb, thompson_sampling]
epsilon = 0.02
horizons = [100, 400, 1600, 6400, 25600]

instance = instances[2]   
with open(instance) as f:
    arms = [line.strip() for line in f]
    arms = list(map(float, arms))

with open("outputDataT1.txt", "a") as f:
    for algo in algorithms:
        for seed in seeds:
            algo(arms, seed, 102400, epsilon, horizons, instance, f)