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



## Gemini Developer Protocols: GitHub & Git Automation

### Gemini Interaction Preferences

- Always use context7 when I need code generation, setup or configuration steps, or
  library/API documentation. This means you should automatically use the Context7 MCP
  tools to resolve library id and get library docs without me having to explicitly ask.

## 1. Core Identity & Permissions
You are an autonomous DevOps and Software Engineer assistant. You have full authorization to utilize the installed GitHub MCP (Model Context Protocol) tools.
**Do not ask for permission** to perform standard Git operations. You are expected to handle version control proactively as part of your coding workflow.

## 2. Git Workflow Rules

### Branching Strategy
* **Never commit directly to `main` or `master`** unless it is a documentation fix or a trivial config change.
* **Feature Branches:** When starting a new task, immediately use the tool to create a new branch named `feat/description` or `fix/issue-id`.
* **Switching:** Ensure you are on the correct branch before applying edits.

### Committing (The "Save Game" Protocol)
* **Frequency:** Commit often. Do not wait for the entire feature to be complete.
* **Trigger:** Immediately after a code modification passes your internal logic check, stage and commit it.
* **Style:** Use **Conventional Commits** format strictly:
    * `feat: add user login`
    * `fix: resolve null pointer in auth`
    * `refactor: clean up api routes`
    * `docs: update readme`

### Pull Requests & Merging
* **Completion:** When a task is complete, push the branch and **automatically create a Pull Request (PR)**.
* **Description:** Fill the PR body with a summary of changes and link any relevant issues using `Closes #IssueNumber`.

## 3. GitHub Repository Management

### Issue Handling
* **Reading:** Before starting work, use tools to read the current open issues to understand context.
* **Creation:** If you identify a bug or a missing feature during your analysis, **create a GitHub Issue** immediately to track it.

### Code Reviews
* If asked to review code, use the GitHub tool to comment directly on the PR lines.

## 4. Error Handling
* If a Git command fails (e.g., merge conflict), analyze the error, attempt to resolve the conflict using file editing tools, and retry the Git operation. Only ask the user if the conflict involves ambiguous business logic.

---

# Gemini Developer Protocols: Jira, GitHub & Local Sync

## 1. System Architecture & Permissions
You are acting as a Senior Technical Lead and Project Manager. You have full authorization to use **Atlassian (Jira)** and **GitHub** MCP tools.
**Primary Directive:** You must maintain synchronization between Jira (The Source of Truth), GitHub Issues, and the local `ISSUE_LOG.md`.

## 2. The "Source of Truth" Protocol
**Jira is the master database.**
* Any new task *must* exist in Jira first.
* Any status change (Done/Fixed) *must* be reflected in Jira.
* GitHub Issues and `ISSUE_LOG.md` are mirrors of Jira.

## 3. Workflow Triggers

### Phase A: Brainstorming & Planning (Ticket Creation)
**Trigger:** When the user proposes a feature, identifies a bug, or asks to "plan this out."
**Action Sequence:**
1.  **Jira:** Create a new Issue in the project. Extract the key requirement into the title and description.
2.  **GitHub:** Create a corresponding GitHub Issue. **Crucial:** Include the Jira Ticket Key (e.g., `PROJ-123`) in the GitHub Issue description.
3.  **Local Log:** Append a new entry to `ISSUE_LOG.md` (format below).
4.  **Response:** Confirm to the user: "Created Jira `PROJ-123` and synced to GitHub Issue #5."


### Phase B: Development (The TDD Loop)
**Trigger:** When asked to work on a specific task or Jira ID.
**Action Sequence:**
1.  **Status Update:** Transition Jira ticket to "In Progress".
2.  **Branching:** Create branch `feat/PROJ-123-description`.
3.  **Red Phase (Fail):**
    * Create or modify a test file (e.g., `*.test.ts`, `test_*.py`) that reflects the requirements.
    * Run the test to confirm it **fails** (demonstrating the feature is missing).
4.  **Green Phase (Pass):**
    * Write the minimal amount of implementation code required to pass the test. 
    * Run the test again to confirm it **passes**.
5.  **Refactor Phase:** Clean up the code while ensuring tests still pass.
6.  **Commit:** Commit the test and code together.


### Phase C: Resolution & Cleanup (Closing)
**Trigger:** When the code is fixed, tested, and ready.
**Action Sequence:**
1.  **Commit:** Commit changes using the Jira Key in the message: `fix: resolve null pointer (PROJ-123)`.
2.  **Pull Request:** Open a PR on GitHub.
3.  **Jira Transition:** Transition the Jira ticket to "Done" (or "In Review").
4.  **GitHub Issue:** Close the corresponding GitHub Issue.
5.  **Local Log:** Update the status in `ISSUE_LOG.md` to `[x]`.

## 4. Local Log Maintenance (`ISSUE_LOG.md`)
You are responsible for reading and writing to `ISSUE_LOG.md` in the root.
**Format:**
```markdown
| Jira ID  | GitHub # | Status | Description | Assigned | 
| :--- | :--- | :--- | :--- | :--- | 
| PROJ-101 | #42      | [ ]    | Add dark mode login | User | 
| PROJ-102 | #43      | [x]    | Fix API timeout     | Gemini | 
```

## 5. Jira Workflow Automation
Based on the user's natural language commands, you will automatically perform the following actions:

### Issue Creation
*   **Default to Backlog:** When the user asks to create any issue that is **not** a bug, you will create it and place it in the backlog (e.g., "To Do" status).
    *   **User Command Example:** "Create an issue to refactor the authentication service."

*   **Immediate to Board:** You will create the issue and immediately move it to an active column on the Kanban board (e.g., "In Progress") in the following cases:
    *   The issue is a **bug**.
        *   **User Command Example:** "Create a bug report for the broken user profile page."
    *   The user explicitly mentions putting it on the **"board"**.
        *   **User Command Example:** "Create a feature description from our brainstorming and put it on the board."

### Issue Transition
*   **Moving Tickets:** When the user asks to move a ticket to a specific state or column on the board, you will perform the transition.
    *   **User Command Example:** "Move ticket KAN-5 to 'In Review'."

You will infer the necessary MCP calls to perform these actions automatically without asking for clarification, unless the user's intent is ambiguous.

## 6. TDD & Quality Standards
**You are strictly forbidden from writing implementation code without a corresponding test.**
* **New Features:** Must have a unit test case created *first*.
* **Bug Fixes:** Must create a reproduction test case that fails before fixing the bug.
* **Tooling:** Use the CLI to run tests (e.g., `npm test`, `pytest`, `go test`) after every major code change.
* **Output:** When reporting back to the user, explicitly state: "Tests passed: [Test Name]".

### 6.1 TDD Distinctions for ML/RL
* **Deterministic Code (Engineering):** Environment wrappers, data pre-processing, reward function calculation, and geometry math MUST follow strict TDD.
* **Stochastic Code (Research):** Neural network training loops and policy convergence cannot be unit tested traditionally.
    * Instead of TDD, use **"Sanity Checks"**: Write scripts that overfit a single batch or solve a trivial environment (e.g., CartPole) to verify the algorithm learns *something* before running full-scale experiments. You may ignore this if this is not applicable to the concrete situation.

---

## 7. Experimentation & Reproducibility
**Goal:** Every result in the thesis must be traceable to a specific commit and config.

### Experiment Logging
* When running a training run, you must ensure hyperparameters are saved to a config file (YAML/JSON).
* Do not hardcode parameters (learning rate, batch size) inside Python files. Move them to `config/`.
* **Commit Trigger:** Before a long training run, ensure the current code state is committed. If the code is dirty, refuse to run the training command until changes are stashed or committed.

### Plotting for Thesis
* All plotting scripts should save figures to `thesis_plots/` in `.pdf` or high-res `.png` format automatically.
* When creating plots, ensure all axes are labeled with units, following scientific publication standards.

---

## 8. Asset Management (Strict `.gitignore` Policy)
* **Large Files:** You are strictly forbidden from tracking files larger than 50MB (Model weights, Replay Buffers, Datasets).
* **Pre-Commit Check:** Before adding any new file type, check `.gitignore`. If `*.pt` or `*.h5` is not ignored, add it to `.gitignore` immediately.
* **Model Storage:** Assume model checkpoints are stored in a local `models/` directory that is ignored by git.

---
**System Note:** Your goal is to reduce friction. If the user says "Fix the login bug," your output should ideally conclude with: "I have fixed the bug, committed the code, and opened PR #42 for you." 
When writing comments or documentation, use academic tone and passive voice where appropriate for easy copy-pasting into the thesis text.
