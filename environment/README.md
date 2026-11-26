# Software environments

The release keeps analysis and training dependencies separate.

## Training

Create a Python 3.12 environment, install the pinned PyTorch CPU wheel, then install the public training requirements and this repository:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environment/requirements-cpu.txt
python -m pip install -e . --no-deps
```

`requirements-cpu.txt` contains only externally available packages. It has no editable installs, VCS URLs, local paths or dependencies on a separate project package. The experiment-specific implementation and training framework code required by the runners are bundled under `src/`.

## Paper analysis

The manuscript tables and figures use a smaller pinned environment:

```bash
python -m pip install -r environment/requirements-analysis.txt
python -m pip install -e . --no-deps
python scripts/reproduce_paper.py
```

Both dependency files are intentionally explicit release locks rather than snapshots of a developer environment.
