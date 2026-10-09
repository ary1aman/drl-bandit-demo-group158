"""Bandit algorithms used in the DRL Assignment 1 demo (Group 158).

All policies share one interface:
    select(context=None) -> arm index
    update(arm, reward, context=None)
Reference: Bouneffouf & Rish (2019), arXiv:1904.10040, and the cited originals.
"""
import numpy as np


class Random:
    def __init__(self, k, rng, **_):
        self.k, self.rng = k, rng

    def select(self, context=None):
        return int(self.rng.integers(self.k))

    def update(self, arm, reward, context=None):
        pass


class EpsilonGreedy:
    """Epsilon-greedy with sample-average value estimates."""

    def __init__(self, k, rng, eps=0.1, **_):
        self.k, self.rng, self.eps = k, rng, eps
        self.n = np.zeros(k)
        self.q = np.zeros(k)

    def select(self, context=None):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.q))

    def update(self, arm, reward, context=None):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class UCB1:
    """UCB1 (Auer et al., 2002): mean + sqrt(2 ln t / n)."""

    def __init__(self, k, rng, **_):
        self.k = k
        self.n = np.zeros(k)
        self.q = np.zeros(k)
        self.t = 0

    def select(self, context=None):
        self.t += 1
        if (self.n == 0).any():
            return int(np.argmin(self.n))
        return int(np.argmax(self.q + np.sqrt(2 * np.log(self.t) / self.n)))

    def update(self, arm, reward, context=None):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class ThompsonSampling:
    """Beta-Bernoulli Thompson Sampling (Agrawal & Goyal, 2012)."""

    def __init__(self, k, rng, **_):
        self.rng = rng
        self.a = np.ones(k)
        self.b = np.ones(k)

    def select(self, context=None):
        return int(np.argmax(self.rng.beta(self.a, self.b)))

    def update(self, arm, reward, context=None):
        self.a[arm] += reward
        self.b[arm] += 1 - reward


class DiscountedUCB:
    """Discounted UCB (Garivier & Moulines, 2011) for drifting rewards."""

    def __init__(self, k, rng, gamma=0.99, xi=0.6, **_):
        self.k, self.gamma, self.xi = k, gamma, xi
        self.s = np.zeros(k)   # discounted reward sums
        self.n = np.zeros(k)   # discounted pull counts

    def select(self, context=None):
        if (self.n < 1e-9).any():
            return int(np.argmin(self.n))
        nt = self.n.sum()
        bonus = np.sqrt(self.xi * np.log(nt) / self.n)
        return int(np.argmax(self.s / self.n + bonus))

    def update(self, arm, reward, context=None):
        self.s *= self.gamma
        self.n *= self.gamma
        self.s[arm] += reward
        self.n[arm] += 1


class SlidingWindowUCB:
    """Sliding-Window UCB (Garivier & Moulines, 2011); O(1) incremental update."""

    def __init__(self, k, rng, window=400, xi=0.6, **_):
        from collections import deque
        self.k, self.w, self.xi = k, window, xi
        self.hist = deque()
        self.n = np.zeros(k)
        self.s = np.zeros(k)

    def select(self, context=None):
        if (self.n == 0).any():
            return int(np.argmin(self.n))
        bonus = np.sqrt(self.xi * np.log(len(self.hist)) / self.n)
        return int(np.argmax(self.s / self.n + bonus))

    def update(self, arm, reward, context=None):
        self.hist.append((arm, reward))
        self.n[arm] += 1
        self.s[arm] += reward
        if len(self.hist) > self.w:
            a, r = self.hist.popleft()
            self.n[a] -= 1
            self.s[a] -= r


class LinUCB:
    """Disjoint LinUCB (Li et al., 2010): one ridge regression per arm."""

    def __init__(self, k, rng, d=5, alpha=1.0, **_):
        self.k, self.alpha = k, alpha
        self.A = [np.eye(d) for _ in range(k)]
        self.b = [np.zeros(d) for _ in range(k)]

    def select(self, context):
        x = context
        scores = []
        for a in range(self.k):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            scores.append(theta @ x + self.alpha * np.sqrt(x @ Ainv @ x))
        return int(np.argmax(scores))

    def update(self, arm, reward, context):
        x = context
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
