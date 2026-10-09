# Bandit demo: DRL Assignment 1, Group 158 (Problem IV)

Small, reproducible simulations that accompany our presentation on
*A Survey on Practical Applications of Multi-Armed and Contextual Bandits*
(Bouneffouf & Rish, 2019, arXiv:1904.10040).

The survey itself runs no experiments, so these toy simulations are **our own** illustration of
three ideas it discusses: exploration strategies, context (LinUCB) and non-stationarity.
They are not a reproduction of any result from the paper.

## Run
```
pip install -r requirements.txt
python run_experiments.py      # ~30 s; writes charts and results.json to results/
```
All runs are seeded, so the numbers below reproduce exactly.

## Files
| File | What it is |
|---|---|
| `bandits.py` | Random, ε-greedy, UCB1, Thompson Sampling (Beta-Bernoulli), Discounted UCB, Sliding-Window UCB, LinUCB |
| `run_experiments.py` | The three experiments and the plots |
| `results/` | `exp1_stationary.png`, `exp2_contextual.png`, `exp3_nonstationary.png`, `results.json` |

## Experiments and results (mean cumulative regret at the final round; lower is better)
**1. Stationary 10-armed Bernoulli bandit** (T=3000, 100 runs)
Random 854.2 · UCB1 274.5 · ε-greedy (ε=0.1) 158.6 · **Thompson Sampling 73.4**
Thompson Sampling learns fastest; random never learns; UCB1's wide bonus explores the most of the learners.

**2. Contextual linear bandit** (5 arms, 5-dim context, T=3000, 30 runs)
Random 1545.3 · ε-greedy without context 1542.9 · **LinUCB 12.7**
A context-free learner cannot beat random when the best arm depends on the context; LinUCB learns the per-arm linear model and nearly removes regret.

**3. Drift: best arm changes every 500 rounds** (5 arms, T=4000, 100 runs)
Thompson Sampling 982.4 · Discounted UCB (γ=0.996) 367.9 · Sliding-Window UCB (w=400) 339.0 · **UCB1 255.9**
Thompson Sampling suffers most under drift because its posterior locks onto old data (about 3x the regret of the drift-aware methods). Discounted and Sliding-Window UCB cut that by roughly 62-65%.
In this small, large-gap problem plain UCB1 stayed best: its ln(t) bonus keeps re-checking every arm. Drift-aware windows cost a constant amount of extra exploration. We also tried other change periods (200, 400, 800, 1000 rounds), a single switch, a longer horizon, and a matched exploration constant; UCB1 stayed ahead in every one we tried. So the result is specific to this setup and we do not claim the paper's guardrail advice is wrong.

## Parameters
Discounted UCB γ=0.996 and Sliding-Window UCB w=400 follow the Garivier & Moulines (2011) guidance for T=4000 (γ≈1-1/(4√T), w≈2√(T log T)); exploration constant ξ=0.6. LinUCB α=0.5, ridge prior = identity.

## References
- Bouneffouf & Rish (2019), arXiv:1904.10040
- Auer, Cesa-Bianchi & Fischer (2002), UCB1
- Agrawal & Goyal (2012), Thompson Sampling
- Li, Chu, Langford & Schapire (2010), LinUCB
- Garivier & Moulines (2011), Discounted / Sliding-Window UCB
