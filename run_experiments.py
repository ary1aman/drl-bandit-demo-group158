"""Runs the three Group 158 demo experiments and saves charts + results.json.

    python run_experiments.py            # full run (about a minute)
Every experiment is seeded, so results are reproducible.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bandits import (Random, EpsilonGreedy, UCB1, ThompsonSampling,
                     DiscountedUCB, SlidingWindowUCB, LinUCB)

INK, MUT, RED, GRN, GLD, BLU = "#3A2E28", "#695F58", "#B85042", "#56705B", "#C69746", "#4F6D8F"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#DAD3CA", "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#EEE9E1"})
OUT = "results/"
results = {}


def plot_curves(curves, colors, title, fname, ylabel="Cumulative regret", vline=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for (name, (mean, se)), c in zip(curves.items(), colors):
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=name, color=c, lw=2)
        ax.fill_between(x, mean - se, mean + se, color=c, alpha=0.15, lw=0)
    for i, v in enumerate(vline or []):
        ax.axvline(v, color=INK, ls=":", lw=0.9, alpha=0.6)
    if vline:
        ax.text(0.99, 0.03, "dotted lines: best arm changes", transform=ax.transAxes, color=INK, ha="right", fontsize=9)
    ax.set_xlabel("Round"); ax.set_ylabel(ylabel); ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT + fname); plt.close(fig)


def summarize(reg):
    return reg.mean(0), reg.std(0) / np.sqrt(len(reg))


# ---------- Experiment 1: stationary Bernoulli MAB ----------
def exp1(T=3000, runs=100):
    means = np.array([0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70])
    pols = {"Random": Random, "ε-greedy (ε=0.1)": EpsilonGreedy, "UCB1": UCB1, "Thompson Sampling": ThompsonSampling}
    curves, final = {}, {}
    for name, P in pols.items():
        reg = np.zeros((runs, T))
        for r in range(runs):
            rng = np.random.default_rng(1000 + r)
            p = P(len(means), rng)
            gap = []
            for t in range(T):
                a = p.select()
                p.update(a, float(rng.random() < means[a]))
                gap.append(means.max() - means[a])
            reg[r] = np.cumsum(gap)
        curves[name] = summarize(reg); final[name] = round(float(reg[:, -1].mean()), 1)
    plot_curves(curves, [MUT, GLD, GRN, RED], "Experiment 1: stationary 10-armed Bernoulli bandit", "exp1_stationary.png")
    return final


# ---------- Experiment 2: contextual linear bandit ----------
def exp2(T=3000, runs=30, k=5, d=5):
    pols = {"Random": Random, "ε-greedy (no context)": EpsilonGreedy, "LinUCB": LinUCB}
    curves, final = {}, {}
    for name, P in pols.items():
        reg = np.zeros((runs, T))
        for r in range(runs):
            rng = np.random.default_rng(2000 + r)
            theta = rng.normal(size=(k, d)); theta /= np.linalg.norm(theta, axis=1, keepdims=True)
            p = P(k, rng, d=d, alpha=0.5)
            gap = []
            for t in range(T):
                x = rng.normal(size=d); x /= np.linalg.norm(x)
                exp_r = theta @ x
                a = p.select(x)
                p.update(a, exp_r[a] + rng.normal(scale=0.1), x)
                gap.append(exp_r.max() - exp_r[a])
            reg[r] = np.cumsum(gap)
        curves[name] = summarize(reg); final[name] = round(float(reg[:, -1].mean()), 1)
    plot_curves(curves, [MUT, GLD, RED], "Experiment 2: contextual bandit (5 arms, 5-dim context)", "exp2_contextual.png")
    return final


# ---------- Experiment 3: non-stationary (best arm changes every 500 rounds) ----------
def exp3(T=4000, runs=100, period=500):
    base = np.array([0.90, 0.60, 0.50, 0.40, 0.30])
    rs = np.random.default_rng(7)
    perms = [rs.permutation(base) for _ in range(T // period + 1)]
    pols = {"UCB1": (UCB1, {}), "Thompson Sampling": (ThompsonSampling, {}),
            "Discounted UCB (γ=0.996)": (DiscountedUCB, {"gamma": 0.996}),
            "Sliding-Window UCB (w=400)": (SlidingWindowUCB, {"window": 400})}
    curves, final = {}, {}
    for name, (P, kw) in pols.items():
        reg = np.zeros((runs, T))
        for r in range(runs):
            rng = np.random.default_rng(3000 + r)
            p = P(5, rng, **kw)
            gap = []
            for t in range(T):
                m = perms[t // period]
                a = p.select()
                p.update(a, float(rng.random() < m[a]))
                gap.append(m.max() - m[a])
            reg[r] = np.cumsum(gap)
        curves[name] = summarize(reg); final[name] = round(float(reg[:, -1].mean()), 1)
    plot_curves(curves, [BLU, MUT, RED, GRN], "Experiment 3: drift (best arm changes every 500 rounds)",
                "exp3_nonstationary.png", vline=list(range(period, T, period)))
    return final


if __name__ == "__main__":
    results["exp1_stationary_final_regret_T3000"] = exp1()
    print("exp1", results["exp1_stationary_final_regret_T3000"])
    results["exp2_contextual_final_regret_T3000"] = exp2()
    print("exp2", results["exp2_contextual_final_regret_T3000"])
    results["exp3_drift_final_regret_T4000"] = exp3()
    print("exp3", results["exp3_drift_final_regret_T4000"])
    json.dump(results, open(OUT + "results.json", "w"), indent=2, ensure_ascii=False)
