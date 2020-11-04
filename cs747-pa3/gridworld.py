import numpy as np
import matplotlib.pyplot as plt

class WindyGridworld:
    def __init__(self, kings_moves):
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

class Solve:
    def __init__(self, algorithm="sarsa", kings_moves=False):
        self.grid = WindyGridworld(kings_moves)
        self.q_val = np.zeros(self.grid.h, self.grid.w, len(self.grid.actions))

    def sarsa_episode(self):
        time_steps = 0
        epsilon = self.grid.epsilon
        alpha = self.grid.alpha

        state = self.grid.start
        action = np.random.choice(self.grid.actions) if np.random.rand()<=epsilon \
            else np.argmax(self.q_val[state])
        while state != self.grid.end:
            next_state = self.grid.move(state, action)
            next_action = np.random.choice(self.grid.actions) if np.random.rand()<=epsilon \
                else np.argmax(self.q_val[next_state])
            # Action value function Update
            self.q_val[state[0], state[1], action] += \
                alpha*(-1 + self.q_val[next_state[0], next_state[1], next_action] -
                       self.q_val[state[0], state[1], action])
            state = next_state
            action = next_action
            time_steps+=1
        return time_steps

    def q_learning_episode(self):
        time_steps = 0
        epsilon = self.grid.epsilon
        alpha = self.grid.alpha

        state = self.grid.start
        while state != self.grid.end:
            action = np.random.choice(self.grid.actions) if np.random.rand() <= epsilon \
                else np.argmax(self.q_val[state])
            next_state = self.grid.move(state, action)

            # Action value function Update
            self.q_val[state[0], state[1], action] += \
                alpha * (-1 + np.max(self.q_val[next_state]) - self.q_val[state[0], state[1], action])
            state = next_state
            time_steps += 1
        return time_steps

    def expected_sarsa_episode(self):
        time_steps = 0
        epsilon = self.grid.epsilon
        alpha = self.grid.alpha
        na = len(self.grid.actions)

        state = self.grid.start
        while state != self.grid.end:
            action = np.random.choice(self.grid.actions) if np.random.rand() <= epsilon \
                else np.argmax(self.q_val[state])
            next_state = self.grid.move(state, action)

            # Action value function Update
            target = (1-epsilon)*np.max(self.q_val[next_state]) + \
                     (epsilon/na)*np.sum(self.q_val[next_state])
            self.q_val[state[0], state[1], action] += \
                alpha * (-1 + target - self.q_val[state[0], state[1], action])
            state = next_state
            time_steps += 1
        return time_steps




if __name__ == "__main__":
    Solve()