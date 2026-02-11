"""
Manifold Projection Synergy (MPS) Module

Khẩu quyết: "Synergy là sự bất ngờ nằm vuông góc với những gì đã biết."

MPS thực hiện 3 bước:
1. Redundancy Estimator: Dự đoán Joint từ Image + Text
2. Orthogonal Synergy Extraction: Trích xuất phần vuông góc với Redundancy
3. Geometric Distillation: Ép Student học cùng hướng và tỷ lệ với Teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RedundancyEstimator(nn.Module):
    """
    Bước 1: Bộ dự đoán tầm thường (The Redundancy Estimator)
    
    Dự đoán Joint embedding từ Image và Text embeddings một cách máy móc.
    Output là Redundancy - phần thông tin dễ đoán.
    """
    def __init__(self, input_dim: int, hidden_dim: int = None, output_dim: int = None):
        """
        Args:
            input_dim: Chiều của Image + Text embeddings (sau khi concat)
            hidden_dim: Chiều ẩn (mặc định = input_dim)
            output_dim: Chiều output (mặc định = input_dim)
        """
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        
        # MLP để dự đoán Joint từ Image + Text
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Khởi tạo weights
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo weights với Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán Joint embedding từ Image và Text.
        
        Args:
            image_emb: (B, D_img) hoặc (B, D) - Image embedding
            text_emb: (B, D_txt) hoặc (B, D) - Text embedding
        
        Returns:
            redundancy: (B, D) - Dự đoán Joint embedding (Redundancy)
        """
        # Nối Image và Text embeddings
        concat_emb = torch.cat([image_emb, text_emb], dim=-1)  # (B, D_img + D_txt)
        
        # Dự đoán Joint
        redundancy = self.projector(concat_emb)  # (B, D)
        
        return redundancy


class OrthogonalProjector(nn.Module):
    """
    Bước 2: Chiếu trực giao (Orthogonal Projection)
    
    Trích xuất phần của vector vuông góc với một vector khác.
    """
    @staticmethod
    def orthogonal_projection(v: torch.Tensor, u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Chiếu vector v vuông góc với vector u.
        
        Công thức: v_orth = v - (v · u / ||u||²) * u
        
        Args:
            v: (B, D) - Vector cần chiếu
            u: (B, D) - Vector chiếu lên (basis)
            eps: Giá trị epsilon để tránh chia cho 0
        
        Returns:
            v_orth: (B, D) - Phần vuông góc của v với u
        """
        # Normalize u
        u_norm_sq = torch.sum(u ** 2, dim=-1, keepdim=True) + eps  # (B, 1)
        u_normalized = u / torch.sqrt(u_norm_sq)  # (B, D)
        
        # Tính projection coefficient
        proj_coeff = torch.sum(v * u_normalized, dim=-1, keepdim=True)  # (B, 1)
        
        # Chiếu vuông góc
        v_orth = v - proj_coeff * u_normalized  # (B, D)
        
        return v_orth
    
    @staticmethod
    def forward(joint_emb: torch.Tensor, redundancy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Trích xuất Synergy từ Joint và Redundancy.
        
        Args:
            joint_emb: (B, D) - Joint embedding thực tế
            redundancy: (B, D) - Redundancy (dự đoán)
            eps: Giá trị epsilon
        
        Returns:
            synergy: (B, D) - Synergy (phần vuông góc)
        """
        return OrthogonalProjector.orthogonal_projection(joint_emb, redundancy, eps)


class MPSModule(nn.Module):
    """
    Manifold Projection Synergy Module
    
    Tổng hợp 3 bước của MPS:
    1. Redundancy Estimator
    2. Orthogonal Synergy Extraction  
    3. Output Synergy vector
    """
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        joint_dim: int,
        hidden_dim: int = None,
        freeze_redundancy_estimator: bool = False
    ):
        """
        Args:
            image_dim: Chiều của Image embedding
            text_dim: Chiều của Text embedding
            joint_dim: Chiều của Joint embedding
            hidden_dim: Chiều ẩn cho Redundancy Estimator
            freeze_redundancy_estimator: Có đóng băng Redundancy Estimator không (dùng cho Teacher)
        """
        super().__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.joint_dim = joint_dim
        self.hidden_dim = hidden_dim or max(image_dim, text_dim, joint_dim)
        
        # Redundancy Estimator
        self.redundancy_estimator = RedundancyEstimator(
            input_dim=image_dim + text_dim,
            hidden_dim=self.hidden_dim,
            output_dim=joint_dim
        )
        
        # Đóng băng nếu cần (cho Teacher)
        if freeze_redundancy_estimator:
            for param in self.redundancy_estimator.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        joint_emb: torch.Tensor,
        normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tính toán Synergy từ Image, Text, và Joint embeddings.
        
        Args:
            image_emb: (B, D_img) - Image embedding
            text_emb: (B, D_txt) - Text embedding
            joint_emb: (B, D_joint) - Joint embedding
            normalize: Có normalize output không
        
        Returns:
            synergy: (B, D_joint) - Synergy vector
            redundancy: (B, D_joint) - Redundancy vector
        """
        # Bước 1: Dự đoán Redundancy
        redundancy = self.redundancy_estimator(image_emb, text_emb)  # (B, D_joint)
        
        # Bước 2: Trích xuất Synergy (vuông góc với Redundancy)
        synergy = OrthogonalProjector.forward(joint_emb, redundancy)  # (B, D_joint)
        
        # Normalize nếu cần
        if normalize:
            synergy = F.normalize(synergy, p=2, dim=-1)
            redundancy = F.normalize(redundancy, p=2, dim=-1)
        
        return synergy, redundancy
    
    def compute_orthogonality(self, synergy: torch.Tensor, redundancy: torch.Tensor) -> torch.Tensor:
        """
        Tính độ vuông góc giữa Synergy và Redundancy.
        
        Args:
            synergy: (B, D) - Synergy vector
            redundancy: (B, D) - Redundancy vector
        
        Returns:
            orthogonality_loss: Scalar - Loss về độ vuông góc (càng gần 0 càng tốt)
        """
        # Cosine similarity giữa Synergy và Redundancy (nên = 0)
        cos_sim = F.cosine_similarity(synergy, redundancy, dim=-1)  # (B,)
        
        # Loss = |cos_sim| (càng gần 0 càng tốt)
        orthogonality_loss = torch.abs(cos_sim).mean()
        
        return orthogonality_loss

    def compute_recon_loss(
        self,
        redundancy: torch.Tensor,
        joint_emb: torch.Tensor,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """
        Loss tái tạo: ép Redundancy gần với Joint nhất có thể (ở mức độ sơ cấp).
        
        RedundancyEstimator được dạy bởi loss này → nó học "dự đoán" Joint từ Image+Text.
        Phần nó không đoán được (phần dư) chính là Synergy — thông tin tinh túy, khó đoán.
        
        Args:
            redundancy: (B, D) - Output của RedundancyEstimator
            joint_emb: (B, D) - Joint embedding thực tế (target)
            loss_type: "mse" hoặc "cosine"
        
        Returns:
            recon_loss: Scalar - Loss tái tạo (càng nhỏ càng tốt)
        """
        if loss_type == "mse":
            return F.mse_loss(redundancy, joint_emb)
        elif loss_type == "cosine":
            # 1 - cos_sim → minimize để redundancy cùng hướng với joint
            cos_sim = F.cosine_similarity(redundancy, joint_emb, dim=-1).mean()
            return 1.0 - cos_sim
        else:
            raise ValueError(f"loss_type must be 'mse' or 'cosine', got {loss_type}")
