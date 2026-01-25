import json
import time
import os
import math
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from src.model.modality_gated_pooling import ModalityGatedPooling
def init_dummy_uniform_gate(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.01)
        nn.init.constant_(m.bias, 0.0)
hidden_dim = 8192
device = "cuda"

teacher_pool_v_qry = ModalityGatedPooling(hidden_dim).to(device)
teacher_pool_t_qry = ModalityGatedPooling(hidden_dim).to(device)
teacher_pool_v_pos = ModalityGatedPooling(hidden_dim).to(device)
teacher_pool_t_pos = ModalityGatedPooling(hidden_dim).to(device)

for m in [
    teacher_pool_v_qry,
    teacher_pool_t_qry,
    teacher_pool_v_pos,
    teacher_pool_t_pos,
]:
    m.apply(init_dummy_uniform_gate)
torch.save(
    {
        "vision_pool_qry": teacher_pool_v_qry.state_dict(),
        "text_pool_qry": teacher_pool_t_qry.state_dict(),
        "vision_pool_pos": teacher_pool_v_pos.state_dict(),
        "text_pool_pos": teacher_pool_t_pos.state_dict(),
        "hidden_dim": hidden_dim,
    },
    "teacher_stage1_dummy.pt",
)
