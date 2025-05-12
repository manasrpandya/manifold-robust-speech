def add_noise_snr(clean_waveform, snr_db=5.0):
    noise = torch.randn_like(clean_waveform)
    signal_power = clean_waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()
    desired_noise_power = signal_power / (10**(snr_db / 10))
    noise = noise * torch.sqrt(desired_noise_power / (noise_power + 1e-8))
    return clean_waveform + noise

def info_nce_loss(z_clean, z_noisy, temperature=0.07):
    batch_size = z_clean.size(0)
    z = torch.cat([z_clean, z_noisy], dim=0)  # [2B, D]
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2B, 2B]

    # Mask out self-comparisons
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

    # Create labels: each clean i should match noisy i + B
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    sim_matrix = sim_matrix / temperature
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
from sklearn.neighbors import NearestNeighbors

def laplacian_loss(z_clean, k=10):
    # Move to CPU for k-NN computation
    z_np = z_clean.detach().cpu().numpy()

    # Build k-NN graph using cosine distance
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(z_np)
    _, indices = nbrs.kneighbors(z_np)

    B = z_clean.shape[0]
    A = torch.zeros((B, B), dtype=torch.float32, device=z_clean.device)  # CUDA

    for i in range(B):
        for j in indices[i][1:]:  # skip self
            sim = F.cosine_similarity(
                z_clean[i].unsqueeze(0), z_clean[j].unsqueeze(0), dim=1
            )
            A[i, j] = sim
            A[j, i] = sim

    D = torch.diag(A.sum(dim=1))
    L = D - A

    lap_loss = torch.trace(z_clean.T @ L @ z_clean)
    return lap_loss / (B * B)
