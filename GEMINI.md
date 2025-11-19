# Gemini Project: Offline Goal-Conditioned Reinforcement Learning

This document provides an overview of the project "Offline Goal Conditioned Reinforcement Learning with Temporal Distance Representations" and instructions for setting up and running experiments.

## Project Overview

This project is a Python-based implementation of the research paper "Offline Goal Conditioned Reinforcement Learning with Temporal Distance Representations". It uses JAX and Flax for building and training deep reinforcement learning agents. The main algorithm implemented is Temporal Metric Distillation (TMD), and the codebase also includes several baseline algorithms for comparison.

The project is a fork of OGBench and uses its environments and evaluation framework.

**Key Technologies:**
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

## Project Modifications

Ignore the following statement, it is not relevant at the moment - The `DEFAULT_DATASET_DIR` in `tmd-release/ogbench/utils.py` has been changed to `/Users/swolf/.gemini/tmp/d92b96257770d7f3e522e2c7c3fb5d0a4bf8fa3e595d79356f8fac67a38914f7/ogbench_data` to avoid permission errors when downloading datasets on macOS with Seatbelt enabled.

## Gemini Interaction Preferences

- Always use context7 when I need code generation, setup or configuration steps, or
  library/API documentation. This means you should automatically use the Context7 MCP
  tools to resolve library id and get library docs without me having to explicitly ask.