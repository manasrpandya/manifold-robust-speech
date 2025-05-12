from model import *
from dataset_loader import *
from utils import *
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import random
#----------------Visualizing Embedding Space using t-SNE------------
def extract_embeddings_with_labels(model, dataset, num_samples=200):
    model.eval()
    all_clean = []
    all_noisy = []
    labels = []

    with torch.no_grad():
        for i in range(num_samples):
            clean, noisy = dataset[i]
            label = dataset.file_list[i].parent.name  # speaker ID
            labels.append(label)
            clean = clean.unsqueeze(0).to(device)
            noisy = noisy.unsqueeze(0).to(device)
            z_clean = model(clean)
            z_noisy = model(noisy)
            all_clean.append(z_clean.cpu())
            all_noisy.append(z_noisy.cpu())

    clean_embed = torch.cat(all_clean, dim=0)
    noisy_embed = torch.cat(all_noisy, dim=0)
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return clean_embed, noisy_embed, encoded_labels

clean_embed, noisy_embed, labels = extract_embeddings_with_labels(model, dataset, num_samples=200)

# Combine and reduce
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
combined = torch.cat([clean_embed, noisy_embed], dim=0)
tsne_result = tsne.fit_transform(combined.numpy())

plt.figure(figsize=(10, 6))
plt.scatter(tsne_result[:200, 0], tsne_result[:200, 1], c=labels, cmap='tab20', label='Clean', alpha=0.6)
plt.scatter(tsne_result[200:, 0], tsne_result[200:, 1], c=labels, cmap='tab20', marker='x', label='Noisy', alpha=0.6)
plt.legend()
plt.title("t-SNE of Clean and Noisy Embeddings (Colored by Speaker ID)")
plt.grid(True)
plt.show()

#--------------------Linear Probe Evaluation---------------
# Generates embeddings with:
# - use_projected: True/False (projected or encoder)
# - concat: if True, returns [encoder || projection]
# - noisy: if True, applies variable-SNR noise
def generate_probe_dataset(dataset, model, use_projected=True, noisy=False, concat=False, num_samples=500):
    X, y = [], []
    model.eval()
    snr_choices = [10, 5, 0, -5]  # for variable SNR

    for i in range(num_samples):
        spec, _ = dataset[i]
        label = dataset.file_list[i].parent.name

        if noisy:
            snr = random.choice(snr_choices)
            x = add_noise_snr(spec, snr_db=snr)
        else:
            x = spec

        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            enc = model.encoder(x)
            if enc.ndim > 2:
                enc = torch.flatten(enc, start_dim=1)  # shape: [1, C*H*W]
            proj = model(x)

        if concat:
            embedding = torch.cat([enc, proj], dim=1)
        else:
            embedding = proj if use_projected else enc

        X.append(embedding.cpu().squeeze().detach().numpy())
        y.append(label)

    return X, y

# === Helper: PCA before classifier
def train_and_eval_logreg(X_train, y_train, X_test, y_test, use_pca=True, n_components=64):
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc

# === Projected: Clean → Clean
X_train_clean_proj, y_train = generate_probe_dataset(dataset, model, use_projected=True, noisy=False, num_samples=500)
X_test_clean_proj, y_test = generate_probe_dataset(dataset, model, use_projected=True, noisy=False, num_samples=200)
acc_clean_proj = train_and_eval_logreg(X_train_clean_proj, y_train, X_test_clean_proj, y_test)

# === Projected: Noisy → Noisy
X_train_noisy_proj, y_train = generate_probe_dataset(dataset, model, use_projected=True, noisy=True, num_samples=500)
X_test_noisy_proj, y_test = generate_probe_dataset(dataset, model, use_projected=True, noisy=True, num_samples=200)
acc_noisy_proj = train_and_eval_logreg(X_train_noisy_proj, y_train, X_test_noisy_proj, y_test)

# === Encoder: Noisy → Noisy
X_train_noisy_enc, y_train = generate_probe_dataset(dataset, model, use_projected=False, noisy=True, num_samples=500)
X_test_noisy_enc, y_test = generate_probe_dataset(dataset, model, use_projected=False, noisy=True, num_samples=200)
acc_noisy_enc = train_and_eval_logreg(X_train_noisy_enc, y_train, X_test_noisy_enc, y_test)

# === Concat: Noisy → Noisy
X_train_concat, y_train = generate_probe_dataset(dataset, model, noisy=True, concat=True, num_samples=500)
X_test_concat, y_test = generate_probe_dataset(dataset, model, noisy=True, concat=True, num_samples=200)
acc_noisy_concat = train_and_eval_logreg(X_train_concat, y_train, X_test_concat, y_test)

# === Print all results
print(f"[Projected] Clean → Clean Probe Accuracy: {acc_clean_proj * 100:.2f}%")
print(f"[Projected] Noisy → Noisy Probe Accuracy: {acc_noisy_proj * 100:.2f}%")
print(f"[Encoder]   Noisy → Noisy Probe Accuracy: {acc_noisy_enc * 100:.2f}%")
print(f"[Concat]    Noisy → Noisy Probe Accuracy: {acc_noisy_concat * 100:.2f}%")

#R---------Robustness Testing Across SNR Levels------------
def evaluate_snr_stability_detailed(model, dataset, snr_list=[10, 5, 0, -5], num_samples=200):
    model.eval()
    results = {}

    with torch.no_grad():
        for snr in snr_list:
            similarities = []
            for i in range(num_samples):
                clean, _ = dataset[i]
                noisy = add_noise_snr(clean, snr_db=snr)

                z_clean = model(clean.unsqueeze(0).to(device))
                z_noisy = model(noisy.unsqueeze(0).to(device))

                sim = F.cosine_similarity(z_clean, z_noisy).item()
                similarities.append(sim)

            results[snr] = similarities  # store full list

    return results

snr_scores_detailed = evaluate_snr_stability_detailed(model, dataset)
avg_scores = {snr: sum(sims)/len(sims) for snr, sims in snr_scores_detailed.items()}

plt.figure(figsize=(6, 4))
plt.plot(list(avg_scores.keys()), list(avg_scores.values()), marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Avg Cosine Similarity (Clean vs Noisy)")
plt.title("SNR Robustness of Embeddings")
plt.grid(True)
plt.show()
