from src.criterions.contrastive_kd_loss import ContrastiveKDLoss
from src.criterions.contrastive_loss import ContrastiveLoss
from src.criterions.contrastive_er_loss import ContrastiveERLoss
from src.criterions.contrastive_loss_2 import ContrastiveLoss2
from src.criterions.eigen_rank_align import ERAlign
from src.criterions.vision_RKD import VisionRKDLoss
from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .emo_loss import EMOLoss
from .em_kd import EMKDLoss
from .em_kd_llava_ov import EMKDLLavaLoss
from .span_propose import SpanProposeCriterion
from .span_propose_attn import SpanProposeCriterionWeighted
from .span_propose_attn_only_phrase import SpanProposeCriterionWeightedOnlyPhrase
from .penultimate_mse_loss import PenultimateMSELoss
from .vision_encoder_kd_loss import VisionEncoderLoss
from .kl_cosine_distill import kl_cosine_distill
from .compute_effective_rank import EffectiveRankLoss
from .contrastive_pooling_loss import ContrastivePoolingLoss
from .eos_attention_kl_loss import EOSAttentionKLLoss
from .eos_attention_kl_intra_cosine_loss import EOSAttentionKLIntraCosineLoss
from .eos_er_combined_loss import EOSERCombinedLoss

criterion_list = {
    "contrastive": ContrastiveLoss,
    "contrastive_2": ContrastiveLoss2,
    "contrastive_er": ContrastiveERLoss,
    "contrastive_rkd": ContrastiveLossWithRKD,
    "proposal_dtw": ProposalLossWithDTW,
    "universal_logit": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "emo_loss": EMOLoss,
    "em_kd": EMKDLoss,
    "em_kd_llava_ov": EMKDLLavaLoss,
    "span_propose": SpanProposeCriterion,
    "span_propose_attn": SpanProposeCriterionWeighted,
    "span_propose_attn_only_phrase": SpanProposeCriterionWeightedOnlyPhrase,

    "vision_rkd": VisionRKDLoss,
    "penultimate_mse": PenultimateMSELoss,
    "contrastive_kd": ContrastiveKDLoss,
    "vision_encoder_kd": VisionEncoderLoss,
    "kl_cosine_distill_loss": kl_cosine_distill,
    "effective_rank_loss": EffectiveRankLoss,
    "eigen_rank_align_loss": ERAlign,
    "contrastive_pooling_loss": ContrastivePoolingLoss,
    "eos_attention_kl_loss": EOSAttentionKLLoss,
    "eos_attention_kl_intra_cosine_loss": EOSAttentionKLIntraCosineLoss,
    "eos_er_combined_loss": EOSERCombinedLoss,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)
