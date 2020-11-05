import numpy as np
class WindyGridworld:
    def __init__(self, kings_moves, stochastic):
        # Gridworld parameters
        self.h, self.w = 7, 10
        self.kings_moves = kings_moves
        if not kings_moves:
            self.actions = list(range(4))  # 0-N, 1-S, 2-E, 3-W
        else:
            self.actions = list(range(8))  # 4-NE, 5-SE, 6-SW, 7-NW

        self.start = (3, 0)
        self.end = (3, 7)

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.epsilon = 0.1
        self.alpha = 0.5

        # define moves
        self.F = []
        if stochastic:
            d = lambda k: bool(self.wind[k])*np.random.choice([-1, 0, 1])
            self.F.append(lambda i, j: (min(max(i - 1 - self.wind[j] + d(j), 0), self.h - 1), j))  # N
            self.F.append(lambda i, j: (max(min(i + 1 - self.wind[j] + d(j), self.h - 1), 0), j))  # S
            self.F.append(lambda i, j: (min(max(i - self.wind[j] + d(j), 0), self.h - 1), min(j + 1, self.w - 1)))  # E
            self.F.append(lambda i, j: (min(max(i - self.wind[j] + d(j), 0), self.h - 1), max(j - 1, 0)))  # W
            self.F.append(lambda i, j: (min(max(i - 1 - self.wind[j] + d(j), 0), self.h - 1),
                                        min(j + 1, self.w - 1)))  # NE
            self.F.append(lambda i, j: (max(min(i + 1 - self.wind[j] + d(j), self.h - 1), 0),
                                        min(j + 1, self.w - 1)))  # SE
            self.F.append(lambda i, j: (max(min(i + 1 - self.wind[j] + d(j), self.h - 1), 0),
                                        max(j - 1, 0)))  # SW
            self.F.append(lambda i, j: (min(max(i - 1 - self.wind[j] + d(j), 0), self.h - 1),
                                        max(j - 1, 0)))  # NW
        else:
            self.F.append(lambda i, j: (max(i - 1 - self.wind[j], 0), j))  # N
            self.F.append(lambda i, j: (max(min(i + 1 - self.wind[j], self.h - 1), 0), j))  # S
            self.F.append(lambda i, j: (max(i - self.wind[j], 0), min(j + 1, self.w - 1)))  # E
            self.F.append(lambda i, j: (max(i - self.wind[j], 0), max(j - 1, 0)))  # W
            self.F.append(lambda i, j: (max(i - 1 - self.wind[j], 0), min(j + 1, self.w - 1)))  # NE
            self.F.append(lambda i, j: (max(min(i + 1 - self.wind[j], self.h - 1), 0),
                                        min(j + 1, self.w - 1)))  # SE
            self.F.append(lambda i, j: (max(min(i + 1 - self.wind[j], self.h - 1), 0),
                                        max(j - 1, 0)))  # SW
            self.F.append(lambda i, j: (max(i - 1 - self.wind[j], 0), max(j - 1, 0)))  # NW

    def move(self, state, action):
        """Function to give next state of agent in windy gridworld upon taking action."""
        return self.F[action](*state)