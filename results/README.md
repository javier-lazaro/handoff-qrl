# Saved experimental results

`results/` contains the saved experimental evidence distributed with the manuscript:

```text
results/
├── lunarlander/
│   ├── manifest.json
│   └── data/
└── minatar/
    ├── manifest.json
    └── data/
```

Both environments expose the same eight-file public data schema. The manifests record the environment role, seed set, row counts and SHA-256 inventory for those files.

`results/` contains source evidence, not derived statistical tables. Deterministic RQ1/RQ2/RQ3/P3 and cross-environment analysis outputs live under `analysis/tables/` and are regenerated from these saved results.

Large execution artefacts such as checkpoints and KD arrays are not distributed. Fresh runs from the public training interface write under `runs/reproduction/` and never modify the saved evidence.

To validate the saved evidence and regenerate the manuscript figures, tables and reported numbers:

```bash
python scripts/reproduce_paper.py
```
