"""
Manifold Projection Synergy (MPS) Loss

Loss function với 3 thành phần:
1. Cosine Similarity Loss: Ép Student Synergy cùng hướng với Teacher Synergy
2. MSE Magnitude Loss: Ép Student Synergy cùng độ lớn với Teacher Synergy
3. Orthogonality Constraint: Ép Student Synergy vuông góc với Student Redundancy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

from src.model.mps import MPSModule


class MPSLoss(nn.Module):
    """
    Manifold Projection Synergy Loss
    
    Áp dụng MPS cho Knowledge Distillation giữa Teacher và Student.
    """
    def __init__(self, args):
        """
        Args:
            args: Training arguments với các thuộc tính:
                - mps_synergy_weight: Trọng số cho synergy loss (default: 1.0)
                - mps_magnitude_weight: Trọng số cho magnitude loss (default: 0.1)
                - mps_orthogonality_weight: Trọng số cho orthogonality constraint (default: 0.5)
                - mps_freeze_teacher_estimator: Đóng băng Teacher's redundancy estimator (default: True)
        """
        super(MPSLoss, self).__init__()
        self.args = args
        
        # Trọng số cho các loss components
        self.synergy_weight = getattr(args, 'mps_synergy_weight', 1.0)
        self.magnitude_weight = getattr(args, 'mps_magnitude_weight', 0.1)
        self.orthogonality_weight = getattr(args, 'mps_orthogonality_weight', 0.5)
        self.recon_weight = getattr(args, 'mps_recon_weight', 0.5)
        self.recon_loss_type = getattr(args, 'mps_recon_loss_type', 'mse')  # "mse" hoặc "cosine"
        self.kd_loss_weight = getattr(args, 'kd_weight', 0.5)
        
        # Lấy dimensions từ args hoặc model
        self.teacher_hidden_dim = getattr(args, 'teacher_hidden_dim', None)
        self.student_hidden_dim = getattr(args, 'student_hidden_dim', None)
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        # 2-epoch MPS: epoch 0 = chỉ recon (train RedundancyEstimator), epoch 1 = không recon (train synergy/magnitude/orth)
        self.current_epoch = None  # None = chế độ cũ (1 epoch, đủ loss)

    def set_epoch(self, epoch: int):
        """
        Dùng cho train 2 epoch:
        - epoch 0: Chỉ dùng mps_recon_loss (train RedundancyEstimator).
        - epoch 1: Tắt mps_recon_loss, train các loss còn lại (RedundancyEstimator đã freeze).
        """
        self.current_epoch = epoch

    def _dist_gather_tensor(self, t: Tensor):
        """Gather tensor từ tất cả các processes trong DDP"""
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def _extract_image_text_embeddings(
        self,
        model,
        joint_input: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trích xuất Image, Text, và Joint embeddings từ model bằng cách sử dụng hidden states.
        
        Args:
            model: MMEBModel instance
            joint_input: Input có cả image và text
        
        Returns:
            image_emb: (B, D) - Image embedding (pooled từ vision tokens)
            text_emb: (B, D) - Text embedding (pooled từ text tokens)
            joint_emb: (B, D) - Joint embedding (pooled từ toàn bộ sequence)
        """
        # Lấy Joint embedding và hidden states
        joint_output = model.encode_input(joint_input)
        
        if isinstance(joint_output, tuple):
            joint_emb = joint_output[0]  # (B, D)
            image_features = joint_output[1]  # List of (num_vision_tokens, D) hoặc None
            hidden_states = joint_output[3] if len(joint_output) > 3 else None  # List of (B, SeqLen, D)
        else:
            joint_emb = joint_output
            image_features = None
            hidden_states = None
        
        batch_size = joint_emb.size(0)
        device = joint_emb.device
        hidden_dim = joint_emb.size(-1)
        
        # Nếu có hidden_states và image_features, trích xuất image và text embeddings
        if hidden_states is not None and image_features is not None and len(hidden_states) > 0:
            from src.model.utils import get_hidden_text_vision, get_hidden_text
            
            last_hidden_state = hidden_states[-1]  # (B, SeqLen, D)
            attention_mask = joint_input.get('attention_mask', None)
            
            # Xác định số lượng text tokens và vision tokens
            VISION_START_TOKEN_ID = 151652
            VISION_END_TOKEN_ID = 151656
            BOS_TOKEN_ID = 151643
            
            input_ids = joint_input['input_ids']
            
            image_embs = []
            text_embs = []
            
            for i in range(batch_size):
                # Đếm số vision tokens từ image_features
                if isinstance(image_features, list):
                    num_vision_tokens = image_features[i].size(0) if i < len(image_features) else 0
                else:
                    num_vision_tokens = image_features.size(1) if image_features.dim() > 2 else 0
                
                # Đếm số text tokens
                text_mask = ((input_ids[i] < VISION_START_TOKEN_ID) | (input_ids[i] > VISION_END_TOKEN_ID)) & \
                           (input_ids[i] != BOS_TOKEN_ID)
                num_text_tokens = text_mask.sum().item()
                
                if num_vision_tokens > 0 and num_text_tokens > 0:
                    # Có cả image và text
                    if attention_mask is not None:
                        text_hidden, vision_hidden = get_hidden_text_vision(
                            last_hidden_state[i],
                            num_text_tokens,
                            num_vision_tokens,
                            attention_mask[i]
                        )
                    else:
                        # Fallback: giả sử không có padding
                        vision_hidden = last_hidden_state[i, :num_vision_tokens, :]
                        text_hidden = last_hidden_state[i, num_vision_tokens:num_vision_tokens + num_text_tokens, :]
                    
                    # Pool vision và text embeddings
                    vision_emb = vision_hidden.mean(dim=0)  # (D,)
                    text_emb = text_hidden.mean(dim=0)  # (D,)
                    
                elif num_vision_tokens > 0:
                    # Chỉ có image
                    if attention_mask is not None:
                        vision_hidden = last_hidden_state[i, :num_vision_tokens, :]
                    else:
                        vision_hidden = last_hidden_state[i, :num_vision_tokens, :]
                    vision_emb = vision_hidden.mean(dim=0)
                    text_emb = torch.zeros(hidden_dim, device=device)
                    
                elif num_text_tokens > 0:
                    # Chỉ có text
                    if attention_mask is not None:
                        text_hidden = get_hidden_text(
                            last_hidden_state[i],
                            num_text_tokens,
                            attention_mask[i]
                        )
                    else:
                        text_hidden = last_hidden_state[i, :num_text_tokens, :]
                    text_emb = text_hidden.mean(dim=0)
                    vision_emb = torch.zeros(hidden_dim, device=device)
                else:
                    # Không có gì
                    vision_emb = torch.zeros(hidden_dim, device=device)
                    text_emb = torch.zeros(hidden_dim, device=device)
                
                image_embs.append(vision_emb)
                text_embs.append(text_emb)
            
            image_emb = torch.stack(image_embs, dim=0)  # (B, D)
            text_emb = torch.stack(text_embs, dim=0)  # (B, D)
            
        else:
            # Fallback: Sử dụng một nửa joint embedding cho image và text
            # Đây là approximation, nhưng vẫn hoạt động được
            image_emb = joint_emb * 0.5
            text_emb = joint_emb * 0.5
        
        return image_emb, text_emb, joint_emb
    
    def _get_synergy_proj(self, distiller, device, dtype):
        """
        Trả về projector để chiếu student synergy (1536) → teacher space (3584).
        Ưu tiên dùng projector đầu tiên của distiller (đã đăng ký, tương thích DeepSpeed/DDP).
        Fallback: tạo Linear riêng và lưu vào distiller để tránh tạo lại mỗi bước.
        """
        if hasattr(distiller, 'projectors') and distiller.projectors is not None:
            projs = distiller.projectors
            if isinstance(projs, nn.ModuleDict):
                first_proj = next(iter(projs.values()))
            else:
                first_proj = projs[0]
            return first_proj

        # Fallback: tạo Linear riêng và đăng ký vào distiller một lần
        if not hasattr(distiller, '_synergy_proj'):
            distiller._synergy_proj = nn.Linear(
                self.student_hidden_dim, self.teacher_hidden_dim, bias=False
            ).to(device=device, dtype=dtype)
        return distiller._synergy_proj

    def forward(self, distiller, input_data):
        """
        Tính MPS Loss.
        
        Args:
            distiller: Distiller instance với student và teacher models
            input_data: Dict chứa:
                - student_inputs: {'qry': ..., 'pos': ...}
                - teacher_inputs: {'qry': ..., 'pos': ...}
        
        Returns:
            loss_dict: Dict chứa các loss components
        """
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        # Lấy inputs
        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)
        device = student_qry_input['input_ids'].device
        
        # Lấy dimensions từ models nếu chưa có
        if self.teacher_hidden_dim is None:
            with torch.no_grad():
                teacher_test = teacher_model.encode_input(teacher_qry_input)
                teacher_emb = teacher_test[0] if isinstance(teacher_test, tuple) else teacher_test
                self.teacher_hidden_dim = teacher_emb.size(-1)
        
        if self.student_hidden_dim is None:
            student_test = student_model.encode_input(student_qry_input)
            student_emb = student_test[0] if isinstance(student_test, tuple) else student_test
            self.student_hidden_dim = student_emb.size(-1)
        
        # Khởi tạo MPS modules nếu chưa có
        if not hasattr(distiller, 'mps_teacher'):
            distiller.mps_teacher = MPSModule(
                image_dim=self.teacher_hidden_dim,
                text_dim=self.teacher_hidden_dim,
                joint_dim=self.teacher_hidden_dim,
                freeze_redundancy_estimator=getattr(self.args, 'mps_freeze_teacher_estimator', True)
            ).to(device)
        
        if not hasattr(distiller, 'mps_student'):
            distiller.mps_student = MPSModule(
                image_dim=self.student_hidden_dim,
                text_dim=self.student_hidden_dim,
                joint_dim=self.student_hidden_dim,
                freeze_redundancy_estimator=False
            ).to(device)
        # Áp dụng phase ngay sau khi mps_student có thể vừa được tạo (2-epoch MPS)
        if getattr(self, 'current_epoch', None) is not None and hasattr(distiller, 'set_mps_phase'):
            distiller.set_mps_phase(self.current_epoch)

        # ========== TEACHER SIDE (Chỉ để lấy mục tiêu) ==========
        with torch.no_grad():
            teacher_model.eval()
            
            # Trích xuất embeddings từ Teacher
            teacher_qry_image_emb, teacher_qry_text_emb, teacher_qry_joint_emb = \
                self._extract_image_text_embeddings(teacher_model, teacher_qry_input)
            
            teacher_pos_image_emb, teacher_pos_text_emb, teacher_pos_joint_emb = \
                self._extract_image_text_embeddings(teacher_model, teacher_pos_input)
            
            # Tính Teacher Synergy
            teacher_qry_synergy, teacher_qry_redundancy = distiller.mps_teacher(
                teacher_qry_image_emb, teacher_qry_text_emb, teacher_qry_joint_emb
            )
            teacher_pos_synergy, teacher_pos_redundancy = distiller.mps_teacher(
                teacher_pos_image_emb, teacher_pos_text_emb, teacher_pos_joint_emb
            )
        
        # ========== STUDENT SIDE (Cần huấn luyện) ==========
        student_model.train()
        
        # Trích xuất embeddings từ Student
        student_qry_image_emb, student_qry_text_emb, student_qry_joint_emb = \
            self._extract_image_text_embeddings(student_model, student_qry_input)
        
        student_pos_image_emb, student_pos_text_emb, student_pos_joint_emb = \
            self._extract_image_text_embeddings(student_model, student_pos_input)
        
        # Tính Student Synergy
        student_qry_synergy, student_qry_redundancy = distiller.mps_student(
            student_qry_image_emb, student_qry_text_emb, student_qry_joint_emb
        )
        student_pos_synergy, student_pos_redundancy = distiller.mps_student(
            student_pos_image_emb, student_pos_text_emb, student_pos_joint_emb
        )
        
        # ========== LOSS CALCULATION ==========

        # Project student synergy → teacher space nếu dims khác nhau
        # (student: 1536, teacher: 3584 — cần cùng space để so sánh cosine / magnitude)
        if self.student_hidden_dim != self.teacher_hidden_dim:
            synergy_proj = self._get_synergy_proj(distiller, device, student_qry_synergy.dtype)
            student_qry_synergy_proj = synergy_proj(student_qry_synergy)
            student_pos_synergy_proj = synergy_proj(student_pos_synergy)
        else:
            student_qry_synergy_proj = student_qry_synergy
            student_pos_synergy_proj = student_pos_synergy

        # Loss 1: Cosine Similarity Loss (cùng hướng)
        qry_cosine_loss = 1 - F.cosine_similarity(
            student_qry_synergy_proj, teacher_qry_synergy, dim=-1
        ).mean()
        pos_cosine_loss = 1 - F.cosine_similarity(
            student_pos_synergy_proj, teacher_pos_synergy, dim=-1
        ).mean()
        synergy_loss = (qry_cosine_loss + pos_cosine_loss) / 2.0
        
        # Loss 2: MSE Magnitude Loss (cùng độ lớn, so sánh sau projection)
        qry_magnitude_loss = F.mse_loss(
            torch.norm(student_qry_synergy_proj, dim=-1),
            torch.norm(teacher_qry_synergy, dim=-1)
        )
        pos_magnitude_loss = F.mse_loss(
            torch.norm(student_pos_synergy_proj, dim=-1),
            torch.norm(teacher_pos_synergy, dim=-1)
        )
        magnitude_loss = (qry_magnitude_loss + pos_magnitude_loss) / 2.0
        
        # Loss 3: Orthogonality Constraint (vuông góc)
        qry_orthogonality_loss = distiller.mps_student.compute_orthogonality(
            student_qry_synergy, student_qry_redundancy
        )
        pos_orthogonality_loss = distiller.mps_student.compute_orthogonality(
            student_pos_synergy, student_pos_redundancy
        )
        orthogonality_loss = (qry_orthogonality_loss + pos_orthogonality_loss) / 2.0
        
        # Loss 4: Reconstruction Loss — dạy RedundancyEstimator
        # Mục tiêu: v_Redundancy ≈ v_Joint (ở mức sơ cấp) → Synergy = phần "khó đoán"
        qry_recon_loss = distiller.mps_student.compute_recon_loss(
            student_qry_redundancy, student_qry_joint_emb, loss_type=self.recon_loss_type
        )
        pos_recon_loss = distiller.mps_student.compute_recon_loss(
            student_pos_redundancy, student_pos_joint_emb, loss_type=self.recon_loss_type
        )
        recon_loss = (qry_recon_loss + pos_recon_loss) / 2.0
        
        # Contrastive Loss (baseline)
        if self.world_size > 1:
            all_student_qry_joint = self._dist_gather_tensor(student_qry_joint_emb)
            all_student_pos_joint = self._dist_gather_tensor(student_pos_joint_emb)
        else:
            all_student_qry_joint = student_qry_joint_emb
            all_student_pos_joint = student_pos_joint_emb

        scores = student_model.compute_similarity(all_student_qry_joint, all_student_pos_joint)
        scores = scores.view(all_student_qry_joint.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_joint.size(0) // all_student_pos_joint.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        mps_total_kd = (
            self.synergy_weight * synergy_loss +
            self.magnitude_weight * magnitude_loss +
            self.orthogonality_weight * orthogonality_loss
        )

        # 2-epoch MPS: epoch 0 chỉ recon, epoch >= 1 không recon
        if self.current_epoch == 0:
            total_loss = self.recon_weight * recon_loss
        else:
            # current_epoch >= 1 hoặc None (chế độ cũ)
            total_loss = (
                contrastive_loss +
                self.kd_loss_weight * mps_total_kd
            )
            if self.current_epoch is None:
                total_loss = total_loss + self.recon_weight * recon_loss

        return {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'mps_recon_loss': recon_loss,
            'mps_synergy_loss': synergy_loss,
            'mps_magnitude_loss': magnitude_loss,
            'mps_orthogonality_loss': orthogonality_loss,
            'mps_total_kd_loss': mps_total_kd,
        }
