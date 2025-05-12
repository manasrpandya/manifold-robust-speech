from dataset_loader import *
from model import *
from utils import *

import time
import torch
import gc
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambdas_to_test = [0.01, 0.1, 1.0]
epochs = 10
temperature = 0.07
results = {}  # Store losses for each lambda

for lambda_lap in lambdas_to_test:
    print(f"\n========== Training with λ = {lambda_lap} ==========\n")
    model = ContrastiveEncoder(projection_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_info_loss = 0.0
        epoch_lap_loss = 0.0
        num_batches = 0

        for batch in loader:
            clean_spec, noisy_spec = batch
            clean_spec = clean_spec.to(device)
            noisy_spec = noisy_spec.to(device)

            optimizer.zero_grad()
            z_clean = model(clean_spec)
            z_noisy = model(noisy_spec)

            loss_info = info_nce_loss(z_clean, z_noisy, temperature)

            k_neighbors = min(10, z_clean.shape[0] - 1)
            loss_lap = laplacian_loss(z_clean, k=k_neighbors)

            total_loss = loss_info + lambda_lap * loss_lap
            total_loss.backward()
            optimizer.step()

            epoch_info_loss += loss_info.item()
            epoch_lap_loss += loss_lap.item()
            num_batches += 1

            # Free memory
            del clean_spec, noisy_spec, z_clean, z_noisy
            gc.collect()
            torch.cuda.empty_cache()

        avg_info = epoch_info_loss / num_batches
        avg_lap = epoch_lap_loss / num_batches
        total = avg_info + lambda_lap * avg_lap

        print(f"Epoch {epoch+1:02d} | λ={lambda_lap} | Total: {total:.4f} | Contrastive: {avg_info:.4f} | Manifold: {avg_lap:.4f}")
        loss_history.append((avg_info, avg_lap))

        # Save model checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), f"your/env/dir/{lambda_lap}_epoch{epoch+1}.pt")#UPDATE DIR

    # Store for later
    results[lambda_lap] = loss_history

    # Save loss history to separate file
    with open(f"/kaggle/working/loss_history_lambda{lambda_lap}.pkl", "wb") as f:
        pickle.dump(loss_history, f)

# Save entire ablation result set
with open("", "wb") as f: #update the dir according to your env
    pickle.dump(results, f)



plt.figure(figsize=(10, 6))

for lambda_val, history in results.items():
    info_losses, lap_losses = zip(*history)
    plt.plot(info_losses, label=f'Contrastive (λ={lambda_val})')
    plt.plot(lap_losses, linestyle='--', label=f'Laplacian (λ={lambda_val})')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves for Different λ Values")
plt.legend()
plt.grid(True)
plt.show()
