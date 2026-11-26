# Handoff training and the attribution problem in hybrid quantum reinforcement learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24566459.svg)](https://doi.org/10.5281/zenodo.24566459)

Code and experimental results for the paper by Javier Lázaro, Juan-Ignacio Vazquez and Pablo Garcia-Bringas.

This repository is self-contained and supports two reproducibility workflows:

1. **Regenerate the paper results.** The saved LunarLander and MinAtar evidence under `results/` reproduces the statistics, tables and figures used in the manuscript.
2. **Rerun the experiments.** The repository includes the training implementation, frozen public protocol files and pinned public dependencies needed to train and evaluate the paper conditions again from scratch.

No access to a separate project repository, unpublished package or original execution system is required.

## Regenerate the paper

Create a Python 3.12 environment and install the lightweight analysis dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements-analysis.txt
python -m pip install -e . --no-deps
python scripts/reproduce_paper.py
```

This validates the saved LunarLander and MinAtar evidence and regenerates all manuscript-facing tables, number macros and figures under `build/paper/`.

## Rerun the experiments

The public training environment is CPU-only. Install the pinned PyTorch CPU wheel and the remaining public dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environment/requirements-cpu.txt
python -m pip install -e . --no-deps
```

For a complete reproduction of an environment, use the matrix launcher:

```bash
python scripts/run_experiments.py \
  --environment minatar \
  --seeds 0-19 \
  --conditions all
```

or replace `minatar` with `lunarlander`.

`--conditions all` is the recommended path because it preserves the dependencies between conditions:

- MinAtar trains one full-classical teacher trajectory per seed and reuses its frozen handoff checkpoints.
- `H(t)` and `C(t)` at `t = 0.25, 0.5, 0.75` reuse the exact same teacher checkpoint and KD dataset bytes.
- LunarLander applies the frozen `51000 + seed` dataset permutation before KD.
- `H(1)` uses the same-seed final full-classical teacher and performs no online quantum PPO.

The public launcher is deliberately serial. Parallel scheduling changes throughput, not the scientific treatment, and is therefore outside the reproduction interface.

Use `--dry-run` to inspect the matrix and `--resume` to continue an interrupted reproduction from completed `PASS` experiments.

### Run one condition

Independent conditions can also be launched directly:

```bash
python scripts/run_experiment.py \
  --environment minatar \
  --condition 'H(0)' \
  --seed 0
```

For `C(0.25)`, `C(0.5)` and `C(0.75)`, first run the matching `H(t)` and provide its output so the control reuses the same teacher and KD data:

```bash
python scripts/run_experiment.py \
  --environment minatar \
  --condition 'C(0.5)' \
  --seed 0 \
  --paired-source runs/reproduction/minatar/h_0p5/seed0
```

A mechanics-only check is available with `--smoke`. Smoke runs use a reduced workload and are **not** paper reproductions.

## Conditions

| Paper label | Legacy label | Meaning |
| --- | --- | --- |
| `H(0)` | Q100 | hybrid VQC head from scratch |
| `H(0.25)` | Q75 | 25% classical training, KD, then 75% hybrid PPO |
| `H(0.5)` | Q50 | 50% classical training, KD, then 50% hybrid PPO |
| `H(0.75)` | Q25 | 75% classical training, KD, then 25% hybrid PPO |
| `H(1)` | Qdownstream | full classical teacher + KD only; no online quantum PPO |
| `C(t)` | classical-delayed | protocol-matched classical twin at the same handoff |
| `full_classical` | classical | full-capacity classical reference |

`C(1)` is not part of the attribution family.

## Software environment

The public training lock contains only externally available dependencies required by this release. Key versions are:

- Python `3.12.13`
- PyTorch `2.12.1` (CPU wheel)
- PennyLane `0.45.1`
- Gymnasium `1.3.0`
- MinAtar `1.0.15`
- NumPy `2.5.0`

All project-specific implementation code required by the runners is included in this repository and installed by `pip install -e . --no-deps`. `environment/requirements-cpu.txt` contains no VCS, editable or machine-local dependencies.

Figure/table regeneration uses the smaller `environment/requirements-analysis.txt` lock (`numpy==2.5.0`, `pandas==3.0.2`, `matplotlib==3.10.8`, `PyYAML==6.0.3`), with their transitive dependencies pinned. Figure PDFs are checked byte-for-byte on the CI image (`ubuntu-24.04`); on other systems, different font files or font-subsetting libraries can change the PDF bytes while the rendered figures stay identical.

The experiments are stochastic. The same seeds, architectures, budgets and dependency versions are supplied, but low-level floating-point differences can make a fresh training trajectory diverge. The saved `results/` data are the numerical evidence used in the manuscript; a fresh run reproduces the frozen protocol rather than promising byte-identical trajectories.

## Saved results

```text
results/lunarlander/     LunarLander evidence used by the manuscript
results/minatar/         MinAtar evidence used by the manuscript
analysis/tables/         derived statistical tables used by the manuscript
```

LunarLander is the development environment: the training recipes were fixed there on a few seeds and then applied unchanged. MinAtar Breakout is the second environment, run under a protocol fixed before its campaign. Both use 20 training seeds in the final paper surface (see the Methods section of the paper).

Machine-readable fields in the manifests, protocol files and derived tables keep the internal role labels of the frozen protocol (`retrospective` for LunarLander, `prospective`/`confirmatory` for MinAtar). They are left unchanged so that the distributed files keep their recorded SHA-256 hashes.

## Repository map

```text
results/                      saved evidence used in the paper
analysis/                     statistical regeneration
figures/                      figure/table/number generators
configs/lunarlander/          public LunarLander protocol
configs/minatar/              public MinAtar protocol
scripts/reproduce_paper.py    regenerate manuscript-facing results
scripts/run_experiment.py     rerun one condition
scripts/run_experiments.py    rerun the protocol-faithful experiment matrix
src/handoff_qrl/              public reproduction interface
src/hybrid_training_qrl/      bundled experiment implementation
src/simplyqrl/                bundled training framework implementation
environment/                  public analysis and training dependency locks
```

For release-scope and data-provenance notes, see [`PROVENANCE.md`](PROVENANCE.md).

## Citation

The archived release of this repository is available on Zenodo: [10.5281/zenodo.24566459](https://doi.org/10.5281/zenodo.24566459). Citation metadata are in [`CITATION.cff`](CITATION.cff).

## Licence

MIT. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
