# Offline Goal-Conditioned Reinforcement Learning with Temporal Distance Representations

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/licenses/by/1.0/)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://tmd-website.github.io/)

<hr style="border: 2px solid gray;"></hr>

This repository contains the Python implementation for the research paper "[Offline Goal Conditioned Reinforcement Learning with Temporal Distance Representations](https://tmd-website.github.io/static/pdf/tmd.pdf)". It is a fork of [OGBench](https://github.com/seohongpark/ogbench) and includes an implementation of the Temporal Metric Distillation (TMD) algorithm, as well as several baseline algorithms for comparison.

## Project Overview

This project provides a framework for research in offline goal-conditioned reinforcement learning. It uses JAX and Flax for building and training deep reinforcement learning agents. The main algorithm implemented is Temporal Metric Distillation (TMD), and the codebase also includes several baseline algorithms for comparison.

### Key Technologies
- Python 3.10+
- JAX
- Flax
- Optax
- `uv` for package management
- `wandb` for logging

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vivekmyers/tmd-release.git
    cd tmd-release
    ```

2.  **Create a virtual environment:**
    It is recommended to use `uv` for creating the virtual environment and installing dependencies, as it is specified in the `pyproject.toml`.

    If you don't have `uv` installed, you can install it with:
    ```bash
    pip install uv
    ```

    Create the virtual environment:
    ```bash
    uv venv
    ```

    Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Install the dependencies from `pyproject.toml`:
    ```bash
    uv pip install -e .
    ```
    This will install all the required packages, including the project itself in editable mode.

## Running Experiments

The main entry point for running experiments is `impls/main.py`. It uses `absl` flags for configuration.

**Example command:**
```bash
python impls/main.py \
    --run_group "TMD_antmaze" \
    --env_name "antmaze-large-navigate-v0" \
    --agent "impls/agents/tmd.py" \
    --train_steps 100000
```

- `--run_group`: A name for the experiment group, used for organizing `wandb` logs.
- `--env_name`: The name of the environment and dataset to use.
- `--agent`: The path to the agent configuration file. The available agents are in the `impls/agents` directory.
- `--train_steps`: The number of training steps.

## Development

- **Agent Implementations:** The implementations of the different reinforcement learning agents are located in the `impls/agents` directory. This includes `tmd.py`, `cmd.py`, `crl.py`, and others.
- **Main Entry Point:** The main script for training and evaluation is `impls/main.py`.
- **Environments:** The environments are based on OGBench and are located in the `ogbench` directory.
- **Formatting:** The project uses `ruff` for code formatting. You can run `ruff format .` to format the code.

## Citation
```bibtex
@article{myers2025offline,
  title={Offline Goal Conditioned Reinforcement Learning with Temporal Distance Representations},
  author={Myers, Vivek and Zheng, Bill Chunyuan and Eysenbach, Benjamin and Levine, Sergey},
  year={2025}
}
```