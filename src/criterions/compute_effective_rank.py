import torch

def compute_effective_rank(tokens, gates=None, eps=1e-6):
    """
    tokens: [B, N, D]
    gates:  [B, N, 1] or None
    """
    if gates is not None:
        tokens = tokens * gates

    # Covariance per sample
    C = torch.matmul(tokens.transpose(-1, -2), tokens) / tokens.size(1)

    trace = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    frob = torch.norm(C, dim=(-2, -1)) ** 2

    return (trace ** 2) / (frob + eps)
