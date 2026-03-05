import numpy as np
import matplotlib.pyplot as plt

def compute_effective_rank(A):
    """Computes the effective rank of a matrix A."""
    # Compute singular values
    S = np.linalg.svd(A, compute_uv=False)
    # Normalize to form a probability distribution
    p = S / np.sum(S)
    # Filter out absolute zeros to avoid log(0) warnings
    p = p[p > 1e-10]
    # Calculate Shannon entropy and effective rank
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

# 1. Setup Data Parameters
N = 100 # Matrix dimensions (100x100)
base_rank = 5

# Generate Synthetic Matrices
# Matrix A: Exact Low Rank
U = np.random.randn(N, base_rank)
V = np.random.randn(base_rank, N)
A_low_rank = np.dot(U, V)

# Matrix B: Low Rank + Moderate Noise
noise_moderate = np.random.randn(N, N) * 0.1
A_noisy = A_low_rank + noise_moderate

# Matrix C: Pure Random Gaussian Noise (Full Rank statistically)
A_random = np.random.randn(N, N)

# 2. Compute Singular Values for Plotting
S_low = np.linalg.svd(A_low_rank, compute_uv=False)
S_noisy = np.linalg.svd(A_noisy, compute_uv=False)
S_random = np.linalg.svd(A_random, compute_uv=False)

# 3. Create the Visualization Dashboard
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Singular Value Spectra ---
axes[0].plot(S_low, label=f'Exact Rank {base_rank}', linewidth=2)
axes[0].plot(S_noisy, label='Rank 5 + Noise', linewidth=2)
axes[0].plot(S_random, label='Pure Noise', linewidth=2)
axes[0].set_yscale('log')
axes[0].set_title('Singular Value Decay (Log Scale)')
axes[0].set_xlabel('Singular Value Index')
axes[0].set_ylabel('Magnitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Algebraic Rank vs Effective Rank ---
matrices = [A_low_rank, A_noisy, A_random]
labels = ['Exact Low Rank', 'Noisy Low Rank', 'Pure Noise']
alg_ranks = [np.linalg.matrix_rank(M) for M in matrices]
eff_ranks = [compute_effective_rank(M) for M in matrices]

x = np.arange(len(labels))
width = 0.35

axes[1].bar(x - width/2, alg_ranks, width, label='Algebraic Rank', color='lightcoral')
axes[1].bar(x + width/2, eff_ranks, width, label='Effective Rank', color='steelblue')
axes[1].set_title('Algebraic vs. Effective Rank')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel('Rank')
axes[1].legend()

# --- Plot 3: Effective Rank vs. Noise Level ---
noise_levels = np.linspace(0, 2, 20)
erank_progression = []

for noise_val in noise_levels:
    temp_noisy_matrix = A_low_rank + np.random.randn(N, N) * noise_val
    erank_progression.append(compute_effective_rank(temp_noisy_matrix))

axes[2].plot(noise_levels, erank_progression, marker='o', color='forestgreen')
axes[2].set_title('Effective Rank as Noise Increases')
axes[2].set_xlabel('Noise Standard Deviation')
axes[2].set_ylabel('Effective Rank')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"visualize_er.png", dpi=300, bbox_inches="tight")
plt.show()