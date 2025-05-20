# Manifold-Aware Contrastive Learning for Noise-Robust Speech Representations

This repository provides the full implementation of the paper **"Manifold-Aware Contrastive Learning for Noise-Robust Speech Representations"**, which proposes a method for learning speech representations that remain robust under varying noise conditions. The method aims to address the limitations of traditional contrastive learning frameworks that rely heavily on augmentative diversity to simulate invariance.

The key innovation lies in introducing a **Laplacian-based graph regularization term** into the contrastive objective. By explicitly preserving the manifold structure of clean data in the latent space, the model enforces geometric alignment between clean and noisy embeddings. This manifold-aware contrastive formulation eliminates the need for engineered augmentation pipelines while still achieving strong robustness in low-SNR scenarios.

Empirical results confirm that this method improves the noise invariance of learned speech embeddings across various evaluation protocols, including t-SNE visualization, linear probing, and performance under SNR degradation. The framework allows for ablation across the regularization strength (denoted λ), revealing its direct impact on geometric consistency and representation collapse.

## Repository Structure

- `dataset_loader.py`: Loads clean-noisy speech pairs and allows on-the-fly noise injection using additive transformations. Handles data partitioning and SNR conditioning.
- `model.py`: Defines the speech encoder backbone and the projection head for contrastive learning. Modular design allows substitution of encoder architecture.
- `train.py`: Primary training loop implementing the manifold-aware contrastive loss function. Includes λ-ablation for tuning graph regularization intensity.
- `evaluate.py`: Scripts for evaluating learned representations through linear probing, 2D t-SNE projection of embedding spaces, and SNR-stratified classification accuracy.
- `utils.py`: Helper functions for graph construction, Laplacian computation, similarity metrics, and custom loss components.

## Installation and Setup
### Note
All data directory references in the code are currently assigned to empty strings (`DIR=""`) and must be manually updated to point to valid local paths before training or evaluation. Replicating results requires specifying correct directories for datasets, saved models, and output logs.

Install dependencies using:

```bash
pip install -r requirements.txt
