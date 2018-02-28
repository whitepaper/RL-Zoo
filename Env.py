import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class Env(object):
    def step(self, a):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class DiscreteEnv(Env):
    def __init__(self, nS, nA, P, isd):
        self.nS = nS  # num of state
        self.nA = nA  # num of action
        self.P = P  # transition probs, P[s][a] == [(prob, s', r, done), ...]
        self.isd = isd  # isd: initial state distribution
        self.reset()

    def step(self, a):
        transitions = self.P[self.s][a]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        p, s, r, d = transitions[i]
        self.s = s
        return (s, r, d, {"prob": p})

    def reset(self):
        self.s = np.random.choice(self.nS, p=self.isd)
        return self.s


class GridWorld(DiscreteEnv):
    def __init__(self, shape=[3, 3], target=[1, 2], wind_prob=.0):
        self.shape = shape
        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        target = target[0] + target[1] * MAX_X

        P = {}
        grid = np.arange(nS).reshape(shape)

        iterator = np.nditer(grid, flags=['multi_index'])
        while not iterator.finished:
            s = iterator.iterindex
            y, x = iterator.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = (lambda s: s == target)
            reward = 0.0 if is_done(s) else - 1.0

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_up_wind = ns_up if y <= 1 else ns_up - MAX_X

                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_right_wind = ns_right if y == 0 else ns_right - MAX_X

                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_down_wind = s

                ns_left = s if x == 0 else s - 1
                ns_left_wind = ns_left if y == 0 else ns_left - MAX_X

                P[s][UP] = [(1.0 - wind_prob, ns_up, reward, is_done(ns_up)),
                            (wind_prob, ns_up_wind, reward, is_done(ns_up_wind))]
                P[s][RIGHT] = [(1.0 - wind_prob, ns_right, reward, is_done(ns_right)),
                               (wind_prob, ns_right_wind, reward, is_done(ns_right_wind))]
                P[s][DOWN] = [(1.0 - wind_prob, ns_down, reward, is_done(ns_down)),
                              (wind_prob, ns_down_wind, reward, is_done(ns_down_wind))]
                P[s][LEFT] = [(1.0 - wind_prob, ns_left, reward, is_done(ns_left)),
                              (wind_prob, ns_left_wind, reward, is_done(ns_left_wind))]

            iterator.iternext()

        isd = np.ones(nS) / (nS - 1)
        isd[target] = 0

        super(GridWorld, self).__init__(nS, nA, P, isd)

    def get_action_name(self, a):
        action2name = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}
        return action2name[a]
