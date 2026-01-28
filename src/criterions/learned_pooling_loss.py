import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F

from src.model.learned_pooling import LearnedPooler
from .utils import get_hidden_text_vision, get_hidden_text

class LearnedPoolingLoss(nn.Module):
    def __init__(self, args):
        super(LearnedPoolingLoss, self).__init__()
        self.args = args
        self.vision_weight = 0.6
        self.reg_weight = 1e-4
        self.q_learn = LearnedPooler(1024)
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
    
    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher
        projectors = distiller.projectors

        if getattr(self, "student_processor", None) is None:
            self.student_processor = distiller.get_student_processor()
        if getattr(self, "teacher_processor", None) is None:
            self.teacher_processor = distiller.get_teacher_processor()

        student_processor = self.student_processor
        teacher_processor = self.teacher_processor
        

        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
        
        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, _, _, student_qry_hidden_states = student_qry_output
        student_pos_reps, _, _, student_pos_hidden_states = student_pos_output
        
        if self.world_size > 1:
            all_student_qry_hidden_states = self._dist_gather_tensor(student_qry_hidden_states)
            all_student_pos_hidden_states = self._dist_gather_tensor(student_pos_hidden_states)
        else:
            all_student_qry_hidden_states = student_qry_hidden_states
            all_student_pos_hidden_states = student_pos_hidden_states
        
        
        q_teacher_qry = self.q_learn(teacher_qry_hidden_states[-1],teacher_qry_input['attention_mask'])
        q_teacher_pos = self.q_learn(teacher_pos_hidden_states[-1],teacher_pos_input['attention_mask'])
        q_student_qry = self.q_learn(all_student_qry_hidden_states[-1],student_qry_input['attention_mask'])
        q_student_pos = self.q_learn(all_student_pos_hidden_states[-1],student_pos_input['attention_mask'])

        mse_qry = F.mse_loss(q_student_qry, q_teacher_qry, reduction='mean')
        mse_pos = F.mse_loss(q_student_pos, q_teacher_pos, reduction='mean')


        return {
            'learned_pooling_loss': mse_qry + mse_pos,
        }
        