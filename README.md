# Interpretable Dyna-Q for Maintenance Intervention Timing on NASA CMAPSS FD001

An interpretable, tabular reinforcement-learning benchmark for deciding **when** to
intervene in aircraft-engine maintenance. The task is framed as a small, fully
transparent Markov Decision Process over the NASA CMAPSS FD001 turbofan
degradation dataset, so that every state, action, and reward can be inspected and
reasoned about directly. The goal is to study how model-based planning (Dyna-Q)
compares with model-free methods and a hand-written rule when the objective is to
time maintenance actions well — neither too early (wasteful) nor too late (risking
failure).

## Algorithms Compared

- **Dyna-Q** — tabular Q-learning augmented with model-based planning steps
- **Q-learning** — off-policy temporal-difference control
- **SARSA** — on-policy temporal-difference control
- **Rule-based maintenance policy** — a hand-written interpretable baseline

## Methodology

- **Dataset:** NASA CMAPSS FD001 (turbofan run-to-failure trajectories).
- **Leakage-free split:** an 80/20 train/test split performed **by engine unit**, so
  no engine appears in both the training and held-out sets.
- **Features:** five degradation-sensitive sensors selected from the FD001 sensor set.
- **State representation (24 states):** the discretized cross-product of a
  **health bucket**, a **degradation-rate bucket**, and a binary **anomaly flag**.
- **Actions (5):** `CONTINUE`, `INSPECT`, `MINOR_REPAIR`, `MAJOR_OVERHAUL`, and
  `REPLACE`.
- **Planning-step comparison:** Dyna-Q is evaluated with `n = 0, 5, 10, and 20`
  planning steps to isolate the effect of additional model-based updates.

## Main Findings

- **Q-learning** and the **best Dyna-Q** configurations achieved a final held-out
  reward of **+5.00 with zero failures** in the primary evaluation.
- **Dyna-Q with 10 planning steps** converged faster than ordinary Q-learning.
- Additional planning improved **sample efficiency** but did **not** improve the
  final learned policy.
- **SARSA** and the **rule-based policy** intervened prematurely.
- The **safety override** did not improve the learned policy.
- Sensitivity tests showed that the learned policy was **fragile under harder
  configurations**.

These results describe the behavior of a small interpretable benchmark. This work is
a research and educational study; it is **not** production-ready, and the learned
policy is **not** appropriate for real aircraft maintenance decisions.

## Repository Structure

```
interpretable-dynaq-cmapss-maintenance/
├── README.md
├── .gitignore
├── papers/
│   ├── AI322_AMBenzon_Full_Paper.pdf
│   └── AI322_miniproject.pdf
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── environment.py
    ├── agents.py
    ├── policies.py
    ├── train.py
    ├── evaluate.py
    └── run_experiments.py
```

## Dataset Setup

The NASA CMAPSS dataset is **not** included in this repository and must be downloaded
separately from the NASA Prognostics Center of Excellence (PCoE) data repository.

After downloading, place the FD001 files under `src/CMAPSSData/`:

```
src/CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

The code expects the data at this location (`src/CMAPSSData/`).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

## Running the Experiments

From the repository root:

```bash
python -m src.run_experiments
```

## Papers

- [Full paper](papers/AI322_AMBenzon_Full_Paper.pdf)
- [Mini-project](papers/AI322_miniproject.pdf)

## Author

Anna Marie Benzon

## License

The source code in `src/` is licensed under the [MIT License](LICENSE).

The papers and presentation materials in `papers/` are licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Copyright © 2026 Anna Marie Benzon.

The NASA CMAPSS dataset is not included in this repository and is not covered
by these licenses. Users must obtain the dataset separately and comply with
its original terms.

## Citation

```bibtex
@misc{benzon2026dynaq,
  author = {Benzon, Anna Marie},
  title  = {Interpretable Dyna-Q for Maintenance Intervention Timing on NASA CMAPSS FD001},
  year   = {2026}
}
```
