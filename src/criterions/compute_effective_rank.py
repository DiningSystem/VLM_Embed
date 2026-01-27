import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


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
    
    def get_unpadded_hidden(self, hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        '''
        Get unpadded hidden states based on attention mask
        Args:
            hidden_state: tensor, the output hidden states from the model [Seq len, Hidden size]
            attention_mask: tensor, the attention mask indicating valid tokens [Seq len]
        Returns:
            unpadded_hidden_state: tensor, the unpadded hidden states [Valid seq len, Hidden size]
        '''
        valid_indices = attention_mask.nonzero(as_tuple=True)[0] # [Num valid tokens]
        unpadded_hidden_state = hidden_state[valid_indices, :] # [Num valid tokens, Hidden size]
        return unpadded_hidden_state
    
    def compute_effective_rank(self, hidden_state: Tensor, eps: float = 1e-5) -> Tensor:
        """
        Compute the effective rank of the hidden states
        Args:
            hidden_state: Tensor of shape [N, D] 
                        (N = batch size or seq len, D = hidden size)
        Returns:
            effective_rank: scalar Tensor
        """
        # [N, D]
        X = hidden_state

        # 1. Centering (zero-mean)
        X = X - X.mean(dim=0, keepdim=True)

        # 2. Covariance matrix: [D, D]
        N = X.size(0)
        cov_matrix = (X.T @ X) / N

        # 3. Numerical stability (important when N << D)
        D = cov_matrix.size(0)
        cov_matrix = cov_matrix + eps * torch.eye(
            D, device=cov_matrix.device, dtype=cov_matrix.dtype
        )

        # 4. Eigenvalues (symmetric PSD matrix)
        eigenvalues = torch.linalg.eigvalsh(cov_matrix.float())

        # 5. Clamp to avoid log(0)
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)

        # 6. Probability distribution
        prob_dist = eigenvalues / eigenvalues.sum()

        # 7. Entropy & effective rank
        entropy = -torch.sum(prob_dist * torch.log(prob_dist))
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
        
        alpha = distiller.student_hidden_dim / distiller.teacher_hidden_dim

        loss_distill = 0.0

        cur_idx_qry_img = 0
        cur_idx_pos_img = 0

        # student_special_ids = torch.tensor(student_tokenizer.all_special_ids, device=student_qry_input['input_ids'].device)
        # teacher_special_ids = torch.tensor(teacher_tokenizer.all_special_ids, device=teacher_qry_input['input_ids'].device)

        # num_student_text_qry_tokens = (~torch.isin(student_qry_input['input_ids'], 
        #                                            student_special_ids)).sum(dim=1)
        # num_student_text_pos_tokens = (~torch.isin(student_pos_input['input_ids'], 
        #                                            student_special_ids)).sum(dim=1)

        # num_teacher_text_qry_tokens = (~torch.isin(teacher_qry_input['input_ids'], 
        #                                            teacher_special_ids)).sum(dim=1)
        # num_teacher_text_pos_tokens = (~torch.isin(teacher_pos_input['input_ids'], 
        #                                            teacher_special_ids)).sum(dim=1)
        
        for i in range(batch_size):
            # --- Xử lý QUERY Image ---
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                # Kiểm tra index hợp lệ
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    stu_feat = student_qry_image_features[cur_idx_qry_img]
                    tea_feat = teacher_qry_image_features[cur_idx_qry_img]
                    stu_vision_eff_rank = self.compute_effective_rank(stu_feat)
                    tea_vision_eff_rank = self.compute_effective_rank(tea_feat)
                    loss_distill += nn.L1Loss()(stu_vision_eff_rank, tea_vision_eff_rank * alpha)
                    cur_idx_qry_img += 1

            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]
                    tea_feat_pos = teacher_pos_image_features[cur_idx_pos_img]

                    stu_vision_eff_rank = self.compute_effective_rank(stu_feat_pos)
                    tea_vision_eff_rank = self.compute_effective_rank(tea_feat_pos)
                    loss_distill += nn.L1Loss()(stu_vision_eff_rank, tea_vision_eff_rank * alpha)
                    cur_idx_pos_img += 1

        loss_distill = loss_distill / (cur_idx_qry_img + cur_idx_pos_img + 1e-8)


        loss = contrastive_loss + self.kd_loss_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }