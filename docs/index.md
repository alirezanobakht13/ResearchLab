# Welcome to Research Lab

**Research Lab (rlab)** is a toolkit designed to make machine learning research reproducible, structured, and manageable.

It addresses two fundamental challenges:
1.  **Reproducibility of "Dirty" Runs:** Research often involves quick iteration where code is run before being committed. `rlab` automatically captures the exact state (Git SHA + uncommitted patch) of every experiment.
2.  **Structured Research Code:** `rlab` provides a set of functional abstractions (State, Config, Loop) powered by JAX and Equinox to write clean, testable, and modular training loops.

## Key Features

### 1. Tracking Module (`researchlab.tracking`)
*   **Context Manager:** `ExperimentTracker` automatically logs Git state to MLflow.
*   **CLI:** `rlab restore` and `rlab diff` to manage and inspect experiment code.
*   **Dirty State:** Captures staged, unstaged, and untracked files.

### 2. Design Module (`researchlab.design`)
*   **State:** JAX PyTree-based state management.
*   **Config:** Immutable, validated configuration using Pydantic.
*   **Selectors:** Decouple your math (kernels) from your data structure.
*   **Loop:** A robust, reusable orchestrator for training and evaluation.

## Installation

Install the core package (tracking only):
```bash
uv add researchlab
```

Install with design features (JAX/Equinox support):
```bash
uv add researchlab[design]
```
