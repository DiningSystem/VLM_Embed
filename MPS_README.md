# Manifold Projection Synergy (MPS) - Hướng dẫn sử dụng

## Tổng quan

**Manifold Projection Synergy (MPS)** là một phương pháp Knowledge Distillation mới cho VLM2Vec, giúp Student học được cách tư duy phân tách thông tin của Teacher thay vì chỉ bắt chước đáp án.

**Khẩu quyết:** *"Synergy là sự bất ngờ nằm vuông góc với những gì đã biết."*

## Kiến trúc MPS

MPS thực hiện 3 bước:

1. **Redundancy Estimator**: Dự đoán Joint embedding từ Image + Text embeddings một cách máy móc
2. **Orthogonal Synergy Extraction**: Trích xuất phần vuông góc với Redundancy từ Joint embedding
3. **Geometric Distillation**: Ép Student học cùng hướng và tỷ lệ với Teacher

## Cách sử dụng

### 1. Cấu hình Training Arguments

Thêm các tham số sau vào training arguments:

```python
# Trọng số cho các loss components
mps_recon_weight = 0.5             # Trọng số cho reconstruction loss (dạy RedundancyEstimator)
mps_recon_loss_type = "mse"        # "mse" hoặc "cosine"
mps_synergy_weight = 1.0           # Trọng số cho synergy loss (cùng hướng)
mps_magnitude_weight = 0.1         # Trọng số cho magnitude loss (cùng độ lớn)
mps_orthogonality_weight = 0.5     # Trọng số cho orthogonality constraint
mps_freeze_teacher_estimator = True # Đóng băng Teacher's redundancy estimator
mps_lr = None                      # Learning rate cho MPS modules (None = dùng learning_rate chung)
kd_weight = 0.5                    # Trọng số tổng thể cho KD loss
```

### 2. Chọn Loss Type

Trong training arguments, set:

```python
kd_loss_type = "mps_loss"
```

### 3. Chạy Training

MPS sẽ tự động:
- Khởi tạo MPS modules cho Teacher và Student
- Trích xuất Image, Text, và Joint embeddings từ hidden states
- Tính toán Synergy và Redundancy
- Áp dụng 3 loại loss: Cosine Similarity, MSE Magnitude, và Orthogonality Constraint

## Loss Components

MPS Loss bao gồm:

0. **Reconstruction Loss (MSE hoặc Cosine)** — **Dạy RedundancyEstimator**
   - Mục tiêu: \(v_{Redundancy}\) càng giống \(v_{Joint}\) càng tốt (ở mức sơ cấp).
   - RedundancyEstimator học "dự đoán" Joint từ Image+Text → phần nó **không** đoán được chính là Synergy (thông tin tinh túy, khó đoán).
   - `recon_loss = MSE(redundancy, joint_emb)` hoặc `1 - cosine_similarity(redundancy, joint_emb)`
   - Chỉ áp dụng cho **Student** (Teacher estimator có thể bị freeze).

1. **Synergy Loss (Cosine Similarity)**: 
   - Ép Student Synergy cùng hướng với Teacher Synergy
   - `loss_synergy = 1 - cosine_similarity(student_synergy, teacher_synergy)`

2. **Magnitude Loss (MSE)**:
   - Ép Student Synergy cùng độ lớn với Teacher Synergy
   - `loss_magnitude = MSE(||student_synergy||, ||teacher_synergy||)`

3. **Orthogonality Constraint**:
   - Ép Student Synergy vuông góc với Student Redundancy
   - `loss_orthogonality = |cosine_similarity(student_synergy, student_redundancy)|`

**Total Loss:**
```
total_loss = contrastive_loss
           + mps_recon_weight * recon_loss
           + kd_weight * (
                 synergy_weight * synergy_loss +
                 magnitude_weight * magnitude_loss +
                 orthogonality_weight * orthogonality_loss
             )
```

## Cấu trúc Code

- `src/model/mps.py`: MPS Module (RedundancyEstimator, OrthogonalProjector, MPSModule)
- `src/criterions/mps_loss.py`: MPS Loss function
- `src/distiller.py`: Tích hợp MPS vào Distiller (tự động thêm vào optimizer)

## Lợi ích

1. **Dense Embeddings**: Vector chứa 2 tầng thông tin nén vào 1 vector
2. **Disentanglement**: Tách rõ phần Redundancy và Synergy trong quá trình huấn luyện
3. **Geometric Structure**: Ép Student học cấu trúc hình học của Teacher
4. **Self-Supervised**: Không cần gán nhãn đâu là synergy, đâu là redundancy

## Train 2 epoch (khuyến nghị cho MPS)

- **Epoch 1:** Chỉ train **RedundancyEstimator** (chỉ dùng `mps_recon_loss`). Student encoder và projectors bị freeze.
- **Epoch 2:** Đóng băng **RedundancyEstimator**, tắt `mps_recon_loss`; train các loss còn lại (synergy, magnitude, orthogonality, contrastive).

Cách dùng: dùng **file train riêng** `train_distillation_mps.py` (không sửa `train_distillation.py`). Script có sẵn:

```bash
bash scripts/train_distill_mps.sh
```

Script gọi `train_distillation_mps.py` với `--kd_loss_type mps_loss` và `--num_train_epochs 2`. Đầu mỗi epoch, `finetune_mps` gọi `criterion.set_epoch(epoch)` và `distiller.set_mps_phase(epoch)`.

## Lưu ý

- MPS modules sẽ được khởi tạo tự động trong lần forward đầu tiên
- Teacher's redundancy estimator thường được đóng băng (freeze) để giữ nguyên mục tiêu
- Image và Text embeddings được trích xuất từ hidden states của model
- MPS hoạt động tốt nhất với các model có thể trích xuất được hidden states (LLaVA, Qwen2-VL, etc.)

## Ví dụ Training Arguments

```python
training_args = TrainingArguments(
    # ... các args khác ...
    kd_loss_type="mps_loss",
    kd_weight=0.5,
    mps_recon_weight=0.5,           # Dạy RedundancyEstimator: redundancy ≈ joint
    mps_recon_loss_type="mse",      # hoặc "cosine"
    mps_synergy_weight=1.0,
    mps_magnitude_weight=0.1,
    mps_orthogonality_weight=0.5,
    mps_freeze_teacher_estimator=True,
    mps_lr=None,  # Sử dụng learning_rate chung
)
```
