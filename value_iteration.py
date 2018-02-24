import numpy as np

DISCOUNT_FACTOR = 1
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class GridWorld(object):
    def __init__(self, shape=[3, 3], target=[2, 2]):
        self.shape = shape
        self.nS = np.prod(shape)
        self.nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        self.target = target[0] + target[1] * MAX_X

        P = {}
        grid = np.arange(self.nS).reshape(shape)

        iterator = np.nditer(grid, flags=['multi_index'])
        while not iterator.finished:
            s = iterator.iterindex
            y, x = iterator.multi_index

            P[s] = {a: [] for a in range(self.nA)}

            is_done = (lambda s: s == self.target)
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            iterator.iternext()

        self.P = P

    def step(self, a):
        transitions = self.P[self.s][a]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        p, s, r, d = transitions[i]
        self.s = s
        return (s, r, d, {"prob": p})


class Agent:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.nS)

    def next_best_action(self, s, V):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + DISCOUNT_FACTOR * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self):
        THETA = 0.0001
        delta = float("inf")

        while delta > THETA:
            delta = 0
            print(np.reshape(self.V, env.shape))
            for s in range(env.nS):
                best_action, best_action_value = self.next_best_action(s, self.V)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value

        policy = np.zeros(env.nS)
        for s in range(env.nS):
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy[s] = best_action

        return policy


def format_policy(policy):
    policy_str = []
    for idx in range(len(policy)):
        entry = policy[idx]
        if entry == UP:
            policy_str.append('U')
        elif entry == DOWN:
            policy_str.append('D')
        elif entry == LEFT:
            policy_str.append('L')
        else:
            policy_str.append('R')
    return policy_str

env = GridWorld()
agent = Agent(env)
policy = agent.optimize()
print(np.reshape(format_policy(policy), env.shape))