# AI for Speech: Manifold-Aware Contrastive Learning for Noise-Robust Speech Representations

**This repo presents an advanced Artificial Intelligence (AI)/Machine Learning (ML) project focused on robust, self-supervised learning for speech data.**  
This work brings a new approach to making machine-learned speech representations resilient to real-world noise, tackling a fundamental challenge for applications like voice assistants, speech analytics, and communications.

**For my non technical friends:**
- _Think of this as teaching a computer to “listen” more like a human—understanding speech reliably even in very noisy environments—by learning from both clean and corrupted examples._
- _The core methods used here are closely related to the latest advances seen in systems like **wav2vec 2.0** and **SimCLR**, but with a new mathematical twist to make them “noise aware” at a deep level._
- 
Key highlights:
- **AI focus:** Uses state-of-the-art *self-supervised* and *contrastive learning*—modern AI techniques that train models without manual labels.
- **Novelty:** Introduces a geometry-preserving regularization technique (graph Laplacian) to keep the learned features stable and meaningful even when speech is noisy.
- **Impact:** Enables downstream AI systems to be more robust to noise, without depending on large-scale data or complex pretraining.

## Project Overview

The full implementation of the paper **"Manifold-Aware Contrastive Learning for Noise-Robust Speech Representations"**, proposes a method for learning speech representations that remain robust under varying noise conditions. The method aims to address the limitations of traditional contrastive learning frameworks that rely heavily on augmentative diversity to simulate invariance.

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
