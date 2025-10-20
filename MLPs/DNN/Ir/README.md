---
language: "none"
tags:
  - "dp"
  - "DNN"
  - "deepmd"
  - "Ir"
  - "metal"
  - "potential"
license: "MIT"
datasets:
  - "training_data"
  - "validation_data"
metrics:
  - "energy"
  - "force"
  - "virial"
base_model: ""
---

# Model Card for DNN-Ir

This model is a DNN model trained for Iridium (Ir) using DeePMD-kit v2.2.10. It can be used with LAMMPS to simulate the behavior of iridium atoms.

## Model Details

### Model Description

This model is a DNN model specifically trained for Iridium (Ir) atoms. It uses the PolymorphGen-MLPKD and DeePMD-kit frameworks to generate a potential energy surface that accurately describes the interactions between iridium atoms.

- **Model type:** DNN model for Iridium (Ir)
- **License:** MIT

### Model Sources

- **Repository:** [https://github.com/LZYUCL/PolymorphGen-MLPKD]

## Uses

### Direct Use

This model can be directly used with LAMMPS to simulate the behavior of iridium atoms. To use it, you need to have LAMMPS with DeepMD support installed. In your LAMMPS input file, add the following lines:

pair_style deepmd graph-3090.pb
pair_coeff * *


## Bias, Risks, and Limitations

This model is designed specifically for iridium and should not be used for other elements.

### Recommendations

Users should be aware that this model is specifically trained for iridium and should not be used for other elements. It is recommended to validate the model's performance for the specific application before using it in production.

## Training Details

### Training Procedure

#### Training Hyperparameters

- **R_max:** 8.0
- **Selection (sel):** 100
- **Hidden irreps:** [240,240,240,240,240]
- **Learning rate:** exponential decay from 0.001 to 3.51e-8 over 5000 steps
- **Loss function:** energy (start_pref_e=0.01, limit_pref_e=1), force (start_pref_f=1000, limit_pref_f=1), virial (start_pref_v=0.01, limit_pref_v=1)
- **Number of steps:** 5,000,000
- **Batch size:** auto

## Evaluation

### Testing Data, Factors & Metrics

#### Metrics

The model was evaluated using energy, force, and virial metrics.

### Results

The model achieved good accuracy in predicting energy, force, and virial for iridium atoms.

#### Summary

The model performs well for iridium atoms under normal conditions.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [NVIDIA GeForce RTX 3090]
- **Hours used:** [29.8 hours]
- **Cloud Provider:** [Private Infrastructure]
- **Compute Region:** [China]
- **Carbon Emitted:** [4.51 kg CO2eq]

## Technical Specifications

### Model Architecture and Objective

The model uses a message-passing neural network to represent the potential energy surface for iridium atoms. The network architecture includes 128x0e + 128x1o hidden irreps.

### Compute Infrastructure

#### Software

- DeePMD-kit v2.2.10
