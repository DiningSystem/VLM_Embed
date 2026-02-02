import torch


def _effective_rank(H, ridge: float = 1e-4, eps=1e-6):
    """
    Compute effective rank from gated hidden states.

    H: [B, N, D]
    Returns: scalar effective rank
    """
    B, N, D = H.shape

    
    H = H.float()
    Hc = H - H.mean(dim=1, keepdim=True)

    # covariance
    cov = torch.matmul(
        Hc.transpose(1, 2), Hc
    ) / (N - 1)

    # eigenvalues
    eye = torch.eye(
        D,
        device=H.device,
        dtype=H.dtype
    ).unsqueeze(0)  # [1,D,D]

    cov = cov + ridge * eye
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=eps)

    p = eigvals / (eigvals.sum(dim=1, keepdim=True) + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=1)

    return torch.exp(entropy)


def compute_effective_rank_loss(
    H_S_v_g, H_S_t_g,
    H_T_v_g, H_T_t_g,
):
    """
    Alpha-normalized effective-rank matching loss.

    All inputs are gated hidden states:
      [B, N, D]

    Returns:
      scalar rank loss
    """

    # hidden dimensions
    D_S = H_S_v_g.shape[-1]
    D_T = H_T_v_g.shape[-1]

    alpha = D_S / D_T

    # vision
    r_S_v = _effective_rank(H_S_v_g)
    r_T_v = _effective_rank(H_T_v_g)

    # text
    r_S_t = _effective_rank(H_S_t_g)
    r_T_t = _effective_rank(H_T_t_g)

    loss_rank = (
        torch.abs(r_S_v - alpha * r_T_v)
        + torch.abs(r_S_t - alpha * r_T_t)
    )

    return loss_rank
