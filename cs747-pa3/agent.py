import numpy as np
from gridworld import WindyGridworld

class Solve:
    def __init__(self, algorithm="sarsa", kings_moves=False, stochastic=False):
        self.grid = WindyGridworld(kings_moves, stochastic)
        self.q_val = np.zeros((self.grid.h, self.grid.w, len(self.grid.actions)))
        self.res = self.play(algorithm)

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

    def play(self, algorithm):
        episodes = 200

        if algorithm == "sarsa":
            f = self.sarsa_episode
        elif algorithm == "q-learning":
            f = self.q_learning_episode
        elif algorithm == "expected-sarsa":
            f = self.expected_sarsa_episode
        steps = []
        for i in range(episodes):
            steps.append(f())

        return np.cumsum(steps)