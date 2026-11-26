# MinAtar Breakout paper evidence

This directory contains the saved MinAtar Breakout evidence distributed with the paper. MinAtar is the prospective confirmatory environment.

The public data schema is the same as for LunarLander:

- `data/handoff_matrix.csv` — seed-level H(t)/C(t) outcomes and staged endpoints;
- `data/full_classical_matrix.csv` — full-capacity classical reference;
- `data/evaluation_endpoints.csv` — teacher, post-KD and final evaluation summaries where defined;
- `data/evaluation_episodes.csv.gz` — episode-level stochastic evaluation records;
- `data/kd_diagnostics.csv` — selected KD diagnostics for staged conditions;
- `data/quantum_forward_accounting.csv` — KD/online and grad/no-grad quantum-forward accounting;
- `data/training_episode_returns.csv.gz` — training-return curves;
- `data/training_update_metrics.csv.gz` — update-level training metrics.

Coverage is seeds 0–19 for the full classical reference and the final H/C condition set. The full-classical trajectory is stored once and acts as the shared teacher trajectory. For staged H(t)/C(t), training-curve rows contain only post-handoff training; H(1) has no online PPO phase.

Deterministic analysis tables (RQ1, RQ2, RQ3, P3 and replication-prediction summaries) are not source data and live under `analysis/tables/minatar/`.

`data/quantum_forward_accounting.csv` excludes evaluation QNode forwards from training resource totals.

Historical path-valued bookkeeping is not part of the public evidence contract. Its removal does not alter seeds, returns, KD diagnostics, timing/resource measurements, quantum-forward counts or statistical results.

Regenerate and validate the paper-facing analysis with:

```bash
python scripts/reproduce_paper.py
```
