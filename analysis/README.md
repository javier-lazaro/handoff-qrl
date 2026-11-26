# Analysis tables

`analysis/tables/` contains deterministic tables derived from the saved evidence under `results/`.

```text
analysis/tables/
├── lunarlander/   # RQ1, RQ2, RQ3, P3 and replication-prediction tables
├── minatar/       # same five per-environment tables
└── *.csv          # deterministic cross-environment summaries
```

These tables are outputs of the analysis pipeline, not experimental source data. They can be regenerated and checked with:

```bash
python scripts/reproduce_paper.py
```

`software_environment.json` records the lightweight public environment used for analysis regeneration.

LunarLander is the development environment and MinAtar Breakout the second environment, run under a protocol fixed before its campaign; both are analysed under the same statistical contract. Machine-readable fields in the manifests, protocol files and derived tables keep the internal role labels of the frozen protocol (`retrospective` for LunarLander, `prospective`/`confirmatory` for MinAtar). They are left unchanged so that the distributed files keep their recorded SHA-256 hashes. Statistical inference is paired by training seed; nonsignificant H(t)-C(t) differences are not treated as equivalence.
