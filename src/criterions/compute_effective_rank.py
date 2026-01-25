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
    
    def compute_effective_rank(self, hidden_state: Tensor) -> Tensor:
        '''
        Compute the effective rank of the hidden states
        Args:
            hidden_state: tensor, the hidden states [Seq len, Hidden size]
        Returns:
            effective_rank: tensor, the effective rank value
        '''
        # Compute covariance matrix
        cov_matrix = torch.matmul(hidden_state.T, hidden_state) / hidden_state.size(0)  # [Hidden size, Hidden size]
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov_matrix.float())  # [Hidden size]
        # Ensure eigenvalues are non-negative
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)
        # Normalize eigenvalues to form a probability distribution
        prob_dist = eigenvalues / torch.sum(eigenvalues)
        # Compute entropy
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
        # Effective rank is exp(entropy)
        effective_rank = torch.exp(entropy).to(dtype=hidden_state.dtype)

        # 🔥 normalize
        # effective_rank = effective_rank / hidden_state.size(1)
        
        return effective_rank

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
        # cur_idx_qry_img = 0
        # cur_idx_pos_img = 0


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
            student_qry_hidden_states_i = student_qry_hidden_states[-1][i] # (seq_len, hidden_size)
            unpad_student_qry_hidden_states_i = self.get_unpadded_hidden(student_qry_hidden_states_i, student_qry_input['attention_mask'][i])
            effective_rank_student_qry = self.compute_effective_rank(unpad_student_qry_hidden_states_i)

            teacher_qry_hidden_states_i = teacher_qry_hidden_states[-1][i] # (seq_len, hidden_size)
            unpad_teacher_qry_hidden_states_i = self.get_unpadded_hidden(teacher_qry_hidden_states_i, teacher_qry_input['attention_mask'][i])
            effective_rank_teacher_qry = self.compute_effective_rank(unpad_teacher_qry_hidden_states_i)

            student_pos_hidden_states_i = student_pos_hidden_states[-1][i] # (seq_len, hidden_size)
            unpad_student_pos_hidden_states_i = self.get_unpadded_hidden(student_pos_hidden_states_i, student_pos_input['attention_mask'][i])
            effective_rank_student_pos = self.compute_effective_rank(unpad_student_pos_hidden_states_i)

            teacher_pos_hidden_states_i = teacher_pos_hidden_states[-1][i] # (seq_len, hidden_size)
            unpad_teacher_pos_hidden_states_i = self.get_unpadded_hidden(teacher_pos_hidden_states_i, teacher_pos_input['attention_mask'][i])
            effective_rank_teacher_pos = self.compute_effective_rank(unpad_teacher_pos_hidden_states_i)


            loss_distill = loss_distill + (nn.L1Loss()(effective_rank_student_qry, 
                                                     alpha * effective_rank_teacher_qry) +
                             nn.L1Loss()(effective_rank_student_pos, 
                                     alpha * effective_rank_teacher_pos)) 
        
        loss_distill = loss_distill / batch_size
        loss = contrastive_loss + self.kd_loss_weight * loss_distill

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }