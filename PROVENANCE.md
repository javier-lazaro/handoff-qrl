# Public release provenance

This repository is a self-contained reproduction package for *Handoff training and the attribution problem in hybrid quantum reinforcement learning*.

## Paper evidence

`results/lunarlander/` and `results/minatar/` contain the numerical evidence distributed with the paper. Their manifests inventory the files included in the public release and pin their distributed SHA-256 hashes.

Development-only source locations, local execution paths and repository identifiers are not part of the public evidence surface. Where historical result tables contained path-valued bookkeeping fields, those fields are blanked in the release. Scientific fields, including seeds, returns, training budgets, KD diagnostics, quantum-forward counts, timing/resource measurements and statistical outputs, are unchanged.

## Training implementation

All code required by the reproduction runners is bundled under `src/`. A user does not need a separate project repository or unpublished Python package.

The public protocol files are:

- `configs/lunarlander/protocol.yaml`
- `configs/minatar/protocol.yaml`

They record the paper-facing environments, architectures, PPO budgets, handoff points, KD procedure, seed contracts, evaluation procedure, resource-accounting semantics and H(t)/C(t) pairing rules. Development file locations and calibration-workspace lineage are intentionally excluded.

## Software environment

`environment/requirements-cpu.txt` pins the external Python dependencies used by the public training interface. PyTorch 2.12.1 is installed separately from the official CPU wheel index. The lock contains no editable installs, VCS URLs, local paths or project-private package references.

`environment/requirements-analysis.txt` contains the smaller dependency set used to regenerate the manuscript tables and figures.

## Cross-condition reproduction contract

The recommended command is `scripts/run_experiments.py --conditions all`, because the paper conditions are not independent jobs.

For MinAtar, one full-classical PPO trajectory is trained per seed and provides the handoff checkpoints. H(t) uses those checkpoints, and each matched C(t) reuses the exact teacher checkpoint and KD dataset bytes of the corresponding H(t). H(1) uses the final teacher checkpoint and has zero online quantum PPO updates.

For LunarLander, H(0.25), H(0.5) and H(0.75) use the 16/32/48-rollout classical handoffs. KD data use the frozen seed families and the train pool is permuted with `51000 + seed` before KD. Each matched C(t) reuses the exact teacher checkpoint and KD dataset bytes of H(t). H(1) reuses the same-seed final full-classical teacher.

The public matrix launcher is serial. Scheduling and recovery machinery are not scientific treatments and are not part of the release interface.

## Fresh reruns versus saved paper results

A fresh stochastic RL run is not claimed to reproduce a historical trajectory byte-for-byte. Low-level floating-point execution can make seeded trajectories diverge across systems. Timing, energy and emissions measurements are also hardware-dependent.

The saved `results/` files are the numerical evidence used by the manuscript. The training package reproduces the frozen scientific protocol: environments, architectures, budgets, seeds, handoffs, KD, pairing and post-hoc evaluation.

## Manuscript regeneration

`python scripts/reproduce_paper.py` validates the saved evidence, regenerates the statistical tables and rebuilds the manuscript-facing tables, number macros and figures. The generated manuscript artefacts are regression-checked against the frozen paper outputs.

## Deliberately excluded

The public package intentionally excludes development-only material such as exploratory/calibration archives, raw execution directories, checkpoints and KD datasets from the completed campaigns, local machine paths and host identifiers, private VCS lineage, and scheduling/recovery infrastructure.
