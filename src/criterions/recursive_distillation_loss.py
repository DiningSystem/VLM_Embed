import hashlib
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import count_clean_text_tokens, get_hidden_text_vision


class RecursiveDistillationLoss(nn.Module):
    """
    Recursive KD with explicit state recursion:
        X^{k+1} = X^k + f_theta(X^k + e_k)

    In this implementation, f_theta is one *full* pass through the student model
    (`student_model.encode_input`) at each step k.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = getattr(self.args, "kd_weight", 1.0)

        self.num_steps = max(1, int(getattr(self.args, "recursive_num_steps", 6)))
        self.backprop_steps = max(1, int(getattr(self.args, "recursive_backprop_steps", self.num_steps)))
        self.backprop_steps = min(self.backprop_steps, self.num_steps)
        self.mean_weight = float(getattr(self.args, "recursive_mean_weight", 1.0))
        self.cov_weight = float(getattr(self.args, "recursive_cov_weight", 0.1))
        self.contrastive_weight = float(getattr(self.args, "recursive_contrastive_weight", 1.0))
        self.attn_weight = float(getattr(self.args, "recursive_attn_weight", 1.0))

        self.enable_kv_cache = bool(getattr(self.args, "recursive_enable_kv_cache", True))
        self.cache_size = max(1, int(getattr(self.args, "recursive_kv_cache_size", 32)))
        self._teacher_cache = OrderedDict()
        self._student_eval_cache = OrderedDict()
        # 1x1 conv mapping for attention alignment (teacher -> student space)
        self.attn_conv1 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.attn_conv1.weight.fill_(1.0)
        for p in self.attn_conv1.parameters():
            p.requires_grad = False
        self._frozen_unused_projectors = False

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
        return torch.cat(all_tensors, dim=0)

    def _cache_key(self, model_tag: str, input_data: dict) -> str:
        hasher = hashlib.sha1(model_tag.encode("utf-8"))
        for key in ["input_ids", "attention_mask", "image_grid_thw", "image_sizes"]:
            if key not in input_data:
                continue
            val = input_data[key]
            if torch.is_tensor(val):
                hasher.update(str(tuple(val.shape)).encode("utf-8"))
                hasher.update(val.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(val, (list, tuple)):
                hasher.update(str(len(val)).encode("utf-8"))
                for item in val:
                    if torch.is_tensor(item):
                        hasher.update(str(tuple(item.shape)).encode("utf-8"))
                    else:
                        hasher.update(str(item).encode("utf-8"))
        return hasher.hexdigest()

    def _put_cache(self, cache: OrderedDict, key: str, value):
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def _encode_with_cache(
        self,
        model,
        model_tag: str,
        input_data: dict,
        enable_grad: bool,
        output_attentions: bool = True,
    ):
        if not self.enable_kv_cache:
            if enable_grad:
                return model.encode_input(input_data, output_attentions=output_attentions)
            with torch.no_grad():
                return model.encode_input(input_data, output_attentions=output_attentions)

        # only cache teacher always + student in eval mode (no grad path)
        if enable_grad:
            return model.encode_input(input_data, output_attentions=output_attentions)

        cache = self._student_eval_cache if model_tag.startswith("student_eval") else self._teacher_cache
        key = self._cache_key(model_tag, input_data)
        key = f"{key}|attn={int(output_attentions)}"
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        with torch.no_grad():
            output = model.encode_input(input_data, output_attentions=output_attentions)
        self._put_cache(cache, key, output)
        return output

    def _step_indices(self, n_layers: int, k_steps: int):
        # n_layers counts embedding hidden too; use 0..n_layers-1
        max_idx = n_layers - 1
        idxs = [0]
        for k in range(1, k_steps + 1):
            idx = (k * max_idx) // k_steps
            idxs.append(max(1, min(idx, max_idx)))
        return idxs

    def _step_embedding(self, step: int, dim: int, device, dtype):
        # deterministic sinusoidal step embedding, shape (1, 1, dim)
        # NOTE: this is generated on-the-fly (non-trainable), so there is no
        # recursive step-embedding parameter to save into checkpoints.
        half = dim // 2
        if half == 0:
            return torch.zeros(1, 1, dim, device=device, dtype=dtype)

        pos = torch.tensor(float(step), device=device, dtype=torch.float32)
        freq = torch.arange(half, device=device, dtype=torch.float32)
        freq = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * freq / max(1, half - 1))
        angle = pos * freq
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=0)
        if emb.numel() < dim:
            emb = torch.cat([emb, emb.new_zeros(dim - emb.numel())], dim=0)
        emb = emb[:dim].to(dtype=dtype)
        return emb.view(1, 1, dim)

    def _extract_first_layer_text_to_vision(
        self,
        attn_list,
        sample_idx: int,
        n_text: int,
        n_vision: int,
        attn_mask: torch.Tensor,
    ):
        if attn_list is None or len(attn_list) == 0:
            return None

        # first layer: (B, H, S, S)
        first_attn = attn_list[0]
        if first_attn is None or first_attn.dim() != 4 or sample_idx >= first_attn.size(0):
            return None
        first_attn = first_attn[sample_idx]  # (H, S, S)
        left_padding = bool(attn_mask[0] == 0 and attn_mask[-1] == 1)

        if left_padding:
            txt_start = first_attn.shape[-1] - n_text
            vis_start = txt_start - n_vision
            txt_slice = slice(txt_start, txt_start + n_text)
            vis_slice = slice(vis_start, vis_start + n_vision)
        else:
            vis_slice = slice(0, n_vision)
            txt_slice = slice(n_vision, n_vision + n_text)

        return first_attn[:, txt_slice, vis_slice].mean(dim=0)  # (Nt, Nv)

    def _attention_alignment_loss(self, student_attn, teacher_attn):
        if student_attn is None or teacher_attn is None:
            return None

        if student_attn.dim() != 2 or teacher_attn.dim() != 2:
            return None

        # 1x1 conv mapping + shape alignment before MSE
        s = student_attn.unsqueeze(0).unsqueeze(0)  # (1,1,Nt,Nv)
        t = teacher_attn.unsqueeze(0).unsqueeze(0)
        conv_weight = self.attn_conv1.weight.to(device=t.device, dtype=t.dtype)
        t_proj = F.conv2d(t, conv_weight, bias=None)
        t_proj = F.interpolate(t_proj, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(s, t_proj)

    def _covariance(self, z: torch.Tensor):
        if z.numel() == 0:
            return None
        z = z - z.mean(dim=0, keepdim=True)
        return (z.transpose(0, 1) @ z) / max(1, z.size(0))

    def _project_update(self, delta, projector):
        if projector is None:
            return delta
        return projector(delta)

    def _recursive_student_updates(
        self,
        student_model,
        student_input,
        base_output,
        k_steps: int,
        capture_first_step_attn: bool = False,
    ):
        """
        Build recursive student token states:
            X^{k+1} = X^k + f_theta(X^k + e_k)

        Here f_theta is one *full* student forward pass each step.
        """
        _, _, _, hidden_states = base_output
        x_prev = hidden_states[0]  # (B, S, D), proxy for X^0
        updates = []

        student_eval_cache_ok = not student_model.training
        final_reps = base_output[0]
        first_step_attention = None
        first_step_image_features = None
        for k in range(1, k_steps + 1):
            grad_enabled_step = (not student_eval_cache_ok) and (k > k_steps - self.backprop_steps)
            step_tag = f"student_eval_step_{k}" if student_eval_cache_ok else f"student_train_step_{k}"
            step_output = self._encode_with_cache(
                student_model,
                step_tag,
                student_input,
                enable_grad=grad_enabled_step,
                output_attentions=(capture_first_step_attn and k == 1),
            )
            _, step_image_features, step_attention, step_hidden = step_output
            final_reps = step_output[0]
            if capture_first_step_attn and k == 1:
                first_step_attention = step_attention
                first_step_image_features = step_image_features
            f_theta_out = step_hidden[-1]  # one full pass output

            e_k = self._step_embedding(k, f_theta_out.size(-1), f_theta_out.device, f_theta_out.dtype)
            # f_theta(X^k + e_k) -- conditioning via step embedding (broadcast over tokens)
            update = f_theta_out + e_k
            if not grad_enabled_step:
                update = update.detach()
            x_next = x_prev + update
            updates.append(update)
            x_prev = x_next

        return updates, final_reps, first_step_attention, first_step_image_features

    def _update_loss_for_side(
        self,
        student_input,
        teacher_input,
        student_base_output,
        teacher_hidden_states,
        recursive_student_updates,
        student_image_features,
        teacher_image_features,
        student_special_ids,
        teacher_special_ids,
        projectors,
    ):
        _, _, _, student_hidden_states = student_base_output
        projector_t2s = projectors["t2s"] if "t2s" in projectors else None

        k_steps = self.num_steps
        teacher_idx = self._step_indices(len(teacher_hidden_states), k_steps)

        s_text_counts = count_clean_text_tokens(student_input, student_special_ids)
        t_text_counts = count_clean_text_tokens(teacher_input, teacher_special_ids)

        bsz = student_input["input_ids"].size(0)
        total_mean = student_hidden_states[0].new_tensor(0.0)
        total_cov = student_hidden_states[0].new_tensor(0.0)
        denom = 0

        cur_s_img = 0
        cur_t_img = 0

        for i in range(bsz):
            if student_image_features is None or teacher_image_features is None:
                continue
            if cur_s_img >= len(student_image_features) or cur_t_img >= len(teacher_image_features):
                continue

            s_num_vision = int(student_image_features[cur_s_img].size(0))
            t_num_vision = int(teacher_image_features[cur_t_img].size(0))
            s_num_text = int(s_text_counts[i].item())
            t_num_text = int(t_text_counts[i].item())

            s_mask = student_input["attention_mask"][i]
            t_mask = teacher_input["attention_mask"][i]

            for k in range(1, k_steps + 1):
                delta_s = recursive_student_updates[k - 1][i]
                delta_t = teacher_hidden_states[teacher_idx[k]][i] - teacher_hidden_states[teacher_idx[k - 1]][i]

                s_txt, s_img = get_hidden_text_vision(delta_s, s_num_text, s_num_vision, s_mask)
                t_txt, t_img = get_hidden_text_vision(delta_t, t_num_text, t_num_vision, t_mask)

                # student branch stays in student space; only teacher is projected to student space
                z_s_img = self._project_update(s_img, None)
                z_s_txt = self._project_update(s_txt, None)
                z_t_img = self._project_update(t_img, projector_t2s)
                z_t_txt = self._project_update(t_txt, projector_t2s)

                mean_loss = F.mse_loss(z_s_img.mean(dim=0), z_t_img.mean(dim=0))
                mean_loss = mean_loss + F.mse_loss(z_s_txt.mean(dim=0), z_t_txt.mean(dim=0))

                cov_loss = 0.0
                cov_s_img = self._covariance(z_s_img)
                cov_t_img = self._covariance(z_t_img)
                cov_s_txt = self._covariance(z_s_txt)
                cov_t_txt = self._covariance(z_t_txt)
                if cov_s_img is not None and cov_t_img is not None:
                    cov_loss = cov_loss + F.mse_loss(cov_s_img, cov_t_img)
                if cov_s_txt is not None and cov_t_txt is not None:
                    cov_loss = cov_loss + F.mse_loss(cov_s_txt, cov_t_txt)

                wk = float(k) / float(k_steps)
                total_mean = total_mean + wk * mean_loss
                total_cov = total_cov + wk * cov_loss
                denom += 1

            cur_s_img += 1
            cur_t_img += 1

        if denom == 0:
            zero = student_hidden_states[0].new_tensor(0.0)
            return zero, zero
        return total_mean / denom, total_cov / denom

    def _contrastive_similarity_distill(self, student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps, projectors):
        teacher_q = F.normalize(projectors["t2s"](teacher_qry_reps), dim=-1)
        teacher_p = F.normalize(projectors["t2s"](teacher_pos_reps), dim=-1)

        s_sim = F.normalize(student_qry_reps, dim=-1) @ F.normalize(student_pos_reps, dim=-1).transpose(0, 1)
        t_sim = teacher_q @ teacher_p.transpose(0, 1)
        return F.l1_loss(s_sim, t_sim)

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher
        projectors = distiller.projectors

        if (not self._frozen_unused_projectors) and ("s2s" in projectors):
            for p in projectors["s2s"].parameters():
                p.requires_grad = False
            self._frozen_unused_projectors = True
        if student_model.training and (not getattr(self, "_gc_enabled", False)):
            if hasattr(student_model.encoder, "gradient_checkpointing_enable"):
                try:
                    student_model.encoder.gradient_checkpointing_enable()
                    if hasattr(student_model.encoder, "config"):
                        student_model.encoder.config.use_cache = False
                except Exception:
                    pass
            self._gc_enabled = True

        if getattr(self, "student_processor", None) is None:
            self.student_processor = distiller.get_student_processor()
        if getattr(self, "teacher_processor", None) is None:
            self.teacher_processor = distiller.get_teacher_processor()

        student_qry_input = input_data["student_inputs"]["qry"]
        student_pos_input = input_data["student_inputs"]["pos"]
        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]

        teacher_model.eval()
        teacher_qry_output = self._encode_with_cache(
            teacher_model,
            "teacher_qry",
            teacher_qry_input,
            enable_grad=False,
            output_attentions=True,
        )
        teacher_pos_output = self._encode_with_cache(
            teacher_model,
            "teacher_pos",
            teacher_pos_input,
            enable_grad=False,
            output_attentions=False,
        )

        student_eval_cache_ok = not student_model.training
        student_qry_output = self._encode_with_cache(
            student_model,
            "student_eval_qry_base" if student_eval_cache_ok else "student_train_qry_base",
            student_qry_input,
            enable_grad=False,
            output_attentions=True,
        )
        student_pos_output = self._encode_with_cache(
            student_model,
            "student_eval_pos_base" if student_eval_cache_ok else "student_train_pos_base",
            student_pos_input,
            enable_grad=False,
            output_attentions=False,
        )

        teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
        teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
        student_qry_reps, student_qry_image_features, student_qry_attention, _ = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, _ = student_pos_output

        recursive_qry_updates, final_student_qry_reps, _, _ = self._recursive_student_updates(
            student_model, student_qry_input, student_qry_output, self.num_steps, capture_first_step_attn=False
        )
        recursive_pos_updates, final_student_pos_reps, _, _ = self._recursive_student_updates(
            student_model, student_pos_input, student_pos_output, self.num_steps, capture_first_step_attn=False
        )

        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(final_student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(final_student_pos_reps)
        else:
            all_student_qry_reps = final_student_qry_reps
            all_student_pos_reps = final_student_pos_reps

        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = self.loss_fn(scores / distiller.temperature, target)

        student_special_ids = torch.tensor(
            self.student_processor.tokenizer.all_special_ids,
            device=student_qry_input["input_ids"].device,
        )
        teacher_special_ids = torch.tensor(
            self.teacher_processor.tokenizer.all_special_ids,
            device=teacher_qry_input["input_ids"].device,
        )

        mean_qry_loss, cov_qry_loss = self._update_loss_for_side(
            student_qry_input,
            teacher_qry_input,
            student_qry_output,
            teacher_qry_hidden_states,
            recursive_qry_updates,
            student_qry_image_features,
            teacher_qry_image_features,
            student_special_ids,
            teacher_special_ids,
            projectors,
        )
        mean_pos_loss, cov_pos_loss = self._update_loss_for_side(
            student_pos_input,
            teacher_pos_input,
            student_pos_output,
            teacher_pos_hidden_states,
            recursive_pos_updates,
            student_pos_image_features,
            teacher_pos_image_features,
            student_special_ids,
            teacher_special_ids,
            projectors,
        )
        mean_loss = 0.5 * (mean_qry_loss + mean_pos_loss)
        cov_loss = 0.5 * (cov_qry_loss + cov_pos_loss)
        update_loss = self.mean_weight * mean_loss + self.cov_weight * cov_loss

        attn_losses = []
        if student_qry_image_features is not None and teacher_qry_image_features is not None:
            s_text_counts = count_clean_text_tokens(student_qry_input, student_special_ids)
            t_text_counts = count_clean_text_tokens(teacher_qry_input, teacher_special_ids)
            for i in range(student_qry_input["input_ids"].size(0)):
                if i >= len(student_qry_image_features) or i >= len(teacher_qry_image_features):
                    continue
                s_att = self._extract_first_layer_text_to_vision(
                    student_qry_attention,
                    i,
                    int(s_text_counts[i].item()),
                    int(student_qry_image_features[i].size(0)),
                    student_qry_input["attention_mask"][i],
                )
                t_att = self._extract_first_layer_text_to_vision(
                    teacher_qry_attention,
                    i,
                    int(t_text_counts[i].item()),
                    int(teacher_qry_image_features[i].size(0)),
                    teacher_qry_input["attention_mask"][i],
                )
                l = self._attention_alignment_loss(s_att, t_att)
                if l is not None:
                    attn_losses.append(l)

        attn_loss = update_loss.new_tensor(0.0) if len(attn_losses) == 0 else torch.stack(attn_losses).mean()

        contrastive_kd_loss = self._contrastive_similarity_distill(
            final_student_qry_reps,
            final_student_pos_reps,
            teacher_qry_reps,
            teacher_pos_reps,
            projectors,
        )

        kd_loss = update_loss + self.attn_weight * attn_loss + self.contrastive_weight * contrastive_kd_loss
        loss = contrastive_loss + self.kd_loss_weight * kd_loss
        # DDP safety: ensure all trainable projector params participate in graph,
        # even when some branches/weights are disabled for a given batch.
        projector_guard = loss.new_tensor(0.0)
        for p in projectors.parameters():
            if p.requires_grad:
                projector_guard = projector_guard + p.sum() * 0.0
        loss = loss + projector_guard

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
            "recursive_update_loss": update_loss,
            "recursive_mean_loss": mean_loss,
            "recursive_cov_loss": cov_loss,
            "recursive_attn_loss": attn_loss,
            "recursive_contrastive_loss": contrastive_kd_loss,
        }
