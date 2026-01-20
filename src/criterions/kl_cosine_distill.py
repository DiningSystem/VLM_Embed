import torch
import torch.nn.functional as F

def kl_cosine_distill(
    z_S_v, z_S_t,
    z_T_v, z_T_t,
    tau=0.07
):
    """
    All inputs: [B, D]
    """
    def cosine(a, b):
        return F.cosine_similarity(a, b, dim=-1)

    s_T = cosine(z_T_v, z_T_t)
    s_S_vT = cosine(z_S_v, z_T_t.detach())
    s_TvS = cosine(z_T_v.detach(), z_S_t)

    p_T = F.softmax(s_T / tau, dim=0)
    p_S1 = F.softmax(s_S_vT / tau, dim=0)
    p_S2 = F.softmax(s_TvS / tau, dim=0)

    return (
        F.kl_div(p_S1.log(), p_T, reduction="batchmean")
        + F.kl_div(p_S2.log(), p_T, reduction="batchmean")
    )
