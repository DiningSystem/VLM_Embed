import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_hidden_text, get_unpadded_hidden



# =====================================================
# Utilities
# =====================================================

def gather_tensor(t: Tensor, world_size, rank):
    if world_size == 1:
        return t

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(all_tensors, t)

    all_tensors[rank] = t
    return torch.cat(all_tensors, dim=0)


# =====================================================
# EOS Influence Map
# =====================================================

def eos_influence_map(vision_tokens, text_tokens, eos):
    """
    Compute EOS-gradient influence map:
    Returns: [N_img, N_txt]
    """

    # Last layer tokens
    V = vision_tokens   # [B,N,d]
    W = text_tokens     # [B,M,d]


    # Probe vector
    u = F.normalize(eos.detach(), dim=-1)

    # Scalar probe
    z = (u * eos).sum(dim=-1).mean()

    # Backprop
    gV = torch.autograd.grad(
        z, V, retain_graph=True
    )[0]

    gW = torch.autograd.grad(
        z, W, retain_graph=True
    )[0]

    # Importance
    sV = gV.norm(dim=-1)[0]    # [N]
    sW = gW.norm(dim=-1)[0]    # [M]

    M = torch.outer(sV, sW)

    return M / (M.sum() + 1e-8)


# =====================================================
# Soft Top-K
# =====================================================

def soft_topk(M, keep_ratio=0.1):

    k = int(keep_ratio * M.numel())

    thresh = torch.topk(
        M.flatten(), k
    ).values[-1]

    mask = (M >= thresh).float()

    M = M * mask

    return M / (M.sum() + 1e-8)


# =====================================================
# Similarity
# =====================================================

def build_similarity(T, S, tau):

    T = F.normalize(T, dim=-1)
    S = F.normalize(S, dim=-1)

    sim = T @ S.T

    return F.softmax(sim / tau, dim=1)


# =====================================================
# KL Loss
# =====================================================

def kl_div(Q, P):

    Q = Q + 1e-8
    P = P + 1e-8

    return (Q * (Q.log() - P.log())).sum()


# =====================================================
# Main Loss Module
# =====================================================

class ContrastiveDerivativeLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.projectors = nn.ModuleDict({

            "t2s": nn.Linear(1024, 768),

            "proj_TI": nn.Linear(1024, 1024),
            "proj_SI": nn.Linear(768, 1024),

            "proj_TT": nn.Linear(1024, 1024),
            "proj_ST": nn.Linear(768, 1024),
        })

        # Weights
        self.kd_weight = args.kd_weight
        self.eos_weight = args.eos_weight

        # Alignment params
        self.keep_ratio = args.keep_ratio
        self.tau_align = args.tau_align

        # Distributed
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0


    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    def forward(self, distiller, input_data):

        student = distiller.student
        teacher = distiller.teacher
        projectors = distiller.projectors

        temperature = distiller.temperature



        # =================================================
        # Inputs
        # =================================================

        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher

        if getattr(self, "student_processor", None) is None:
            self.student_processor = distiller.get_student_processor()
        if getattr(self, "teacher_processor", None) is None:
            self.teacher_processor = distiller.get_teacher_processor()

        student_processor = self.student_processor
        teacher_processor = self.teacher_processor

        student_tokenizer = student_processor.tokenizer
        teacher_tokenizer = teacher_processor.tokenizer
        

        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
        
        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)

        loss_distill = 0.0

        cur_idx_qry_img = 0
        cur_idx_pos_img = 0

        student_special_ids = torch.tensor(student_tokenizer.all_special_ids, device=student_qry_input['input_ids'].device)
        teacher_special_ids = torch.tensor(teacher_tokenizer.all_special_ids, device=teacher_qry_input['input_ids'].device)

        
        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        num_teacher_text_qry_tokens = count_clean_text_tokens(teacher_qry_input, teacher_special_ids)
        num_teacher_text_pos_tokens = count_clean_text_tokens(teacher_pos_input, teacher_special_ids)
        
        student_qry_text_last = []
        student_qry_vision_first = []
        student_pos_text_last = []
        student_pos_vision_first = []

        teacher_qry_text_last = []
        teacher_qry_vision_first = []
        teacher_pos_text_last = []
        teacher_pos_vision_first = []

        for i in range(batch_size):
            # --- Xử lý QUERY Image ---
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                # Kiểm tra index hợp lệ
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    stu_feat = student_qry_image_features[cur_idx_qry_img]
                    tea_feat = teacher_qry_image_features[cur_idx_qry_img]
                    student_qry_vision_first.append(stu_feat)
                    teacher_qry_vision_first.append(tea_feat)

                    last_stu_text_hidden_state, _ = get_hidden_text_vision(
                        student_qry_hidden_states[-1][i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat.size(0),
                        student_qry_input['attention_mask'][i]
                    )

                    student_qry_text_last.append(last_stu_text_hidden_state)

                    last_tea_text_hidden_state, _ = get_hidden_text_vision(
                        teacher_qry_hidden_states[-1][i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )
                    
                    teacher_qry_text_last.append(last_tea_text_hidden_state)

                    cur_idx_qry_img += 1
            # no vision tokens
            else:
                last_stu_text_hidden_state = get_hidden_text(
                    student_qry_hidden_states[-1][i],
                    num_student_text_qry_tokens[i].item(),
                    student_qry_input['attention_mask'][i]
                )

                last_tea_text_hidden_state = get_hidden_text(
                    teacher_qry_hidden_states[-1][i],
                    num_teacher_text_qry_tokens[i].item(),
                    teacher_qry_input['attention_mask'][i]
                )

                
            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                    student_pos_vision_first.append(stu_feat_pos)
                    teacher_pos_vision_first.append(tea_feat_pos)

                    last_stu_text_hidden_state, _ = get_hidden_text_vision(
                        student_pos_hidden_states[-1][i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )

                    student_pos_text_last.append(last_stu_text_hidden_state)

                    last_tea_text_hidden_state, _ = get_hidden_text_vision(
                        teacher_pos_hidden_states[-1][i],
                        num_teacher_text_pos_tokens[i].item(),
                        tea_feat_pos.size(0),
                        teacher_pos_input['attention_mask'][i]
                    )

                    teacher_pos_text_last.append(last_tea_text_hidden_state)
                    

                    cur_idx_pos_img += 1
            # no vision tokens
            else:
                last_stu_text_hidden_state = get_hidden_text(
                    student_pos_hidden_states[-1][i],
                    num_student_text_pos_tokens[i].item(),
                    student_pos_input['attention_mask'][i]
                )

                last_tea_text_hidden_state = get_hidden_text(
                    teacher_pos_hidden_states[-1][i],
                    num_teacher_text_pos_tokens[i].item(),
                    teacher_pos_input['attention_mask'][i]
                )



        # =================================================
        # 3. EOS Alignment Loss
        # =================================================

        # -------- Teacher Map --------
        student_qry_vision_first = torch.cat(student_qry_vision_first)
        student_qry_text_last = torch.cat(student_qry_text_last)
        student_pos_vision_first = torch.cat(student_pos_vision_first)
        student_pos_text_last = torch.cat(student_pos_text_last)

        teacher_qry_vision_first = torch.cat(teacher_qry_vision_first)
        teacher_qry_text_last = torch.cat(teacher_qry_text_last)
        teacher_pos_vision_first = torch.cat(teacher_pos_vision_first)
        teacher_pos_text_last = torch.cat(teacher_pos_text_last)

        MT_qry = eos_influence_map(
            vision_tokens=teacher_qry_vision_first, text_tokens=teacher_qry_text_last, eos=teacher_qry_reps
        )

        MT_qry = soft_topk(MT_qry, self.keep_ratio)


        # -------- Student Map --------

        PS_qry = eos_influence_map(
            vision_tokens=student_qry_vision_first, text_tokens=student_qry_text_last, eos=student_qry_reps
        )

        PS_qry = F.softmax(
            PS_qry / self.tau_align, dim=None
        )


        # -------- Tokens --------

        VT_qry = teacher_qry_vision_first
        WT_qry = teacher_qry_text_last

        VS_qry = student_qry_vision_first
        WS_qry = student_qry_text_last


        # -------- Projection --------

        VTp_qry = projectors["proj_TI"](VT_qry)
        VSp_qry = projectors["proj_SI"](VS_qry)

        WTp_qry = projectors["proj_TT"](WT_qry)
        WSp_qry = projectors["proj_ST"](WS_qry)


        # -------- Similarities --------

        SI_qry = build_similarity(
            VTp_qry, VSp_qry, self.tau_align
        )

        ST_qry = build_similarity(
            WTp_qry, WSp_qry, self.tau_align
        )


        # -------- Teacher → Student --------

        M_hat_qry = SI_qry.T @ MT_qry @ ST_qry

        M_hat_qry = F.softmax(
            M_hat_qry / self.tau_align, dim=None
        )


        # -------- KL qry --------

        eos_loss_qry = kl_div(M_hat_qry, PS_qry)


        MT_pos = eos_influence_map(
            vision_tokens=teacher_pos_vision_first, text_tokens=teacher_pos_text_last, eos=teacher_pos_reps
        )

        MT_pos = soft_topk(MT_pos, self.keep_ratio)


        # -------- Student Map --------

        PS_pos = eos_influence_map(
            vision_tokens=student_pos_vision_first, text_tokens=student_pos_text_last, eos=student_pos_reps
        )

        PS_pos = F.softmax(
            PS_pos / self.tau_align, dim=None
        )


        # -------- Tokens --------

        VT_pos = teacher_pos_vision_first
        WT_pos = teacher_pos_text_last

        VS_pos = student_pos_vision_first
        WS_pos = student_pos_text_last


        # -------- Projection --------

        VTp_pos = projectors["proj_TI"](VT_pos)
        VSp_pos = projectors["proj_SI"](VS_pos)

        WTp_pos = projectors["proj_TT"](WT_pos)
        WSp_pos = projectors["proj_ST"](WS_pos)


        # -------- Similarities --------

        SI_pos = build_similarity(
            VTp_pos, VSp_pos, self.tau_align
        )

        ST_pos = build_similarity(
            WTp_pos, WSp_pos, self.tau_align
        )


        # -------- Teacher → Student --------

        M_hat_pos = SI_pos.T @ MT_pos @ ST_pos

        M_hat_pos = F.softmax(
            M_hat_pos / self.tau_align, dim=None
        )


        # -------- KL pos --------

        eos_loss_pos = kl_div(M_hat_pos, PS_pos)

        eos_loss = 0.5 * (eos_loss_qry + eos_loss_pos)
        # =================================================
        # 4. Final Loss
        # =================================================

        loss = (
            contrastive_loss
            + self.eos_weight * eos_loss
        )


        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "eos_loss": eos_loss
        }
