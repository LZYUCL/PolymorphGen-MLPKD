# PolymorphGen-MLPKD Framework

A comprehensive computational framework for polymorphic material structure generation, machine learning potential (MLP) training, and knowledge distillation (KD) for efficient property prediction.

## Overview

This framework integrates multiple computational methods to:
- Analyze atomic configurations through entropy-symmetry landscape for thermodynamic dimensionality reduction
- Generate diverse atomic configurations through genetic algorithms in entropy-symmetry space
- Implement knowledge distillation from complex MPNN models to efficient DNN models
- Automate DFT task distribution and dataset generation

## Features

- **Polymorph Structure Analysis**: Physical dimensionality reduction of thermodynamic configurations using entropy-symmetry landscape
- **Polymorph Structure Generation**: Genetic algorithm-based mutation in entropy-symmetry landscape
- **Multi-scale Sampling**: Combines ML-AIMD with targeted genetic mutations
- **Automated DFT Workflow**: High-throughput calculations with Auto-DFT platform
- **Machine Learning Potential**: Integration of MPNN and DNN for accurate force fields
- **Knowledge Distillation**: Transfer learning from MPNN to DNN models

## Prerequisites

- Python 3.7 or higher
- NumPy
- SciPy
- Matplotlib
- ASE (Atomic Simulation Environment)
- DeepMD-kit v3.0.0
- DeepMD-GNN
