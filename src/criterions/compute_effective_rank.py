import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_hidden_text, get_unpadded_hidden

class EffectiveRankLoss(nn.Module):
    def __init__(self, args):
        super(EffectiveRankLoss, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = self.args.kd_weight
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def compute_effective_rank(
        self,
        hidden_state: torch.Tensor, # [N, D]
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Tính toán Effective Rank chuẩn theo bài báo Diff-eRank.
        Sử dụng SVD trên dữ liệu đã được chuẩn hóa để ổn định hơn.
        """
        # 1. Chuyển sang fp32 để đảm bảo độ chính xác cho SVD [cite: 459]
        X = hidden_state.float() 

        # 2. Khử kỳ vọng (Mean Centering) - [cite: 83, 84]
        # z_bar = mean(z_i)
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean

        # 3. Chuẩn hóa L2 (L2 Normalization) - 
        # Bắt buộc: Mỗi vector (z_i - z_bar) phải có độ dài bằng 1
        norms = torch.norm(X_centered, p=2, dim=1, keepdim=True)
        # Tránh chia cho 0
        X_normalized = X_centered / (norms + eps)

        # 4. Tính Singular Values (s) của ma trận X_normalized / sqrt(N)
        # Ma trận hiệp phương sai Sigma = (1/N) * X_norm.T @ X_norm
        # Do đó eigenvalues của Sigma = (singular_values của X_norm / sqrt(N))^2
        N = X_normalized.size(0)
        s = torch.linalg.svdvals(X_normalized) / torch.sqrt(torch.tensor(N))

        # 5. Eigenvalues của ma trận hiệp phương sai [cite: 77, 96]
        eigvals = s * s 

        # 6. Tính Shannon Entropy trên spectrum [cite: 94, 95]
        # Chú ý: Tổng eigvals của ma trận hiệp phương sai từ các vector chuẩn hóa luôn bằng 1 [cite: 91]
        # Nên ta không cần chia eigvals.sum() nữa, nhưng clamp để tránh lỗi log(0)
        prob = eigvals.clamp(min=eps)
        
        # Entropy H(K) = -sum(lambda_i * log(lambda_i)) [cite: 92, 95]
        entropy = -(prob * torch.log(prob)).sum()
        
        # 7. Effective Rank = exp(H) [cite: 87, 90]
        effective_rank = torch.exp(entropy)

        return effective_rank.to(dtype=hidden_state.dtype)

    def forward(self, distiller, input_data):
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
        
        loss_vision_er = 0.0
        loss_last_text_er = 0.0

        for i in range(batch_size):
            # --- Xử lý QUERY Image ---
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                # Kiểm tra index hợp lệ
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    stu_feat = student_qry_image_features[cur_idx_qry_img]
                    tea_feat = teacher_qry_image_features[cur_idx_qry_img]
                    # loss_vision_er += nn.L1Loss()(self.compute_effective_rank(stu_feat), 
                    #                               self.compute_effective_rank(tea_feat) * alpha)

                    # last_stu_text_hidden_state, _ = get_hidden_text_vision(
                    #     student_qry_hidden_states[-1][i],
                    #     num_student_text_qry_tokens[i].item(),
                    #     stu_feat.size(0),
                    #     student_qry_input['attention_mask'][i]
                    # )

                    # last_tea_text_hidden_state, _ = get_hidden_text_vision(
                    #     teacher_qry_hidden_states[-1][i],
                    #     num_teacher_text_qry_tokens[i].item(),
                    #     tea_feat.size(0),
                    #     teacher_qry_input['attention_mask'][i]
                    # )

                    # loss_last_text_er += nn.L1Loss()(
                    #     self.compute_effective_rank(last_stu_text_hidden_state),
                    #     self.compute_effective_rank(last_tea_text_hidden_state)
                    # )

                    last_stu_hidden_state = get_unpadded_hidden(
                        student_qry_hidden_states[-1][i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat.size(0),
                        student_qry_input['attention_mask'][i]
                    )

                    last_tea_hidden_state = get_unpadded_hidden(
                        teacher_qry_hidden_states[-1][i],
                        num_teacher_text_qry_tokens[i].item(),
                        tea_feat.size(0),
                        teacher_qry_input['attention_mask'][i]
                    )

                    alpha1 = last_stu_hidden_state.size(0)
                    alpha2 = last_tea_hidden_state.size(0)

                    loss_distill += nn.L1Loss()(
                        self.compute_effective_rank(last_stu_hidden_state) / alpha1,
                        self.compute_effective_rank(last_tea_hidden_state) / alpha2
                    ) 

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

                # loss_last_text_er += nn.L1Loss()(
                #     self.compute_effective_rank(last_stu_text_hidden_state),
                #     self.compute_effective_rank(last_tea_text_hidden_state)
                # )
                
                alpha1 = last_stu_text_hidden_state.size(0)
                alpha2 = last_tea_text_hidden_state.size(0)

                loss_distill += nn.L1Loss()(
                    self.compute_effective_rank(last_stu_text_hidden_state) / alpha1,
                    self.compute_effective_rank(last_tea_text_hidden_state) / alpha2
                )

            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                    # loss_vision_er += nn.L1Loss()(self.compute_effective_rank(stu_feat_pos), 
                    #                               self.compute_effective_rank(tea_feat_pos))

                    # last_stu_text_hidden_state, _ = get_hidden_text_vision(
                    #     student_pos_hidden_states[-1][i],
                    #     num_student_text_pos_tokens[i].item(),
                    #     stu_feat_pos.size(0),
                    #     student_pos_input['attention_mask'][i]
                    # )

                    # last_tea_text_hidden_state, _ = get_hidden_text_vision(
                    #     teacher_pos_hidden_states[-1][i],
                    #     num_teacher_text_pos_tokens[i].item(),
                    #     tea_feat_pos.size(0),
                    #     teacher_pos_input['attention_mask'][i]
                    # )

                    # loss_last_text_er += nn.L1Loss()(
                    #     self.compute_effective_rank(last_stu_text_hidden_state),
                    #     self.compute_effective_rank(last_tea_text_hidden_state)
                    # )

                    last_stu_hidden_state = get_unpadded_hidden(
                        student_pos_hidden_states[-1][i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )

                    last_tea_hidden_state = get_unpadded_hidden(
                        teacher_pos_hidden_states[-1][i],
                        num_teacher_text_pos_tokens[i].item(),
                        tea_feat_pos.size(0),
                        teacher_pos_input['attention_mask'][i]
                    )

                    alpha1 = last_stu_hidden_state.size(0)
                    alpha2 = last_tea_hidden_state.size(0)

                    loss_distill += nn.L1Loss()(
                        self.compute_effective_rank(last_stu_hidden_state) / alpha1,
                        self.compute_effective_rank(last_tea_hidden_state) / alpha2
                    )

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

                # loss_last_text_er += nn.L1Loss()(
                #     self.compute_effective_rank(last_stu_text_hidden_state),
                #     self.compute_effective_rank(last_tea_text_hidden_state)
                # )

                alpha1 = last_stu_text_hidden_state.size(0)
                alpha2 = last_tea_text_hidden_state.size(0)

                loss_distill += nn.L1Loss()(
                    self.compute_effective_rank(last_stu_text_hidden_state) / alpha1,
                    self.compute_effective_rank(last_tea_text_hidden_state) / alpha2
                )

        # loss_vision_er = loss_vision_er / (cur_idx_qry_img + cur_idx_pos_img + 1e-8)
        # loss_last_text_er = loss_last_text_er / (2*batch_size + 1e-8)
        
        # loss_distill = loss_last_text_er

        loss_distill = loss_distill / (2*batch_size)

        loss = contrastive_loss + self.kd_loss_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }