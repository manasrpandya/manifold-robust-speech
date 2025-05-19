# Manifold-Aware Contrastive Learning for Noise-Robust Speech Representations

This repository contains the full implementation of our paper, where we explore a geometry-preserving approach to robust speech embedding learning. Our method combines SimCLR-style contrastive learning with Laplacian graph-based regularization, without relying on augmentation diversity.

## Structure

- `dataset_loader.py`: Dataset class for clean-noisy pairs with on-the-fly noise injection.
- `model.py`: Encoder and projection head definitions.
- `train.py`: Training script with manifold-aware loss and λ-ablation.
- `evaluate.py`: Probing, t-SNE, and SNR-based robustness evaluation.
- `utils.py`: Shared utilities and loss functions.

## Setup

```bash
pip install -r requirements.txt
