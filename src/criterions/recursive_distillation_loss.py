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
        # Memory-safe default: only backprop through the last recursive step unless
        # explicitly overridden by --recursive_backprop_steps.
        self.backprop_steps = max(1, int(getattr(self.args, "recursive_backprop_steps", 1)))
        self.backprop_steps = min(self.backprop_steps, self.num_steps)
        self.mean_weight = float(getattr(self.args, "recursive_mean_weight", 1.0))
        self.cov_weight = float(getattr(self.args, "recursive_cov_weight", 0.1))
        self.contrastive_weight = float(getattr(self.args, "recursive_contrastive_weight", 1.0))
        self.attn_weight = float(getattr(self.args, "recursive_attn_weight", 1.0))

        self.enable_kv_cache = bool(getattr(self.args, "recursive_enable_kv_cache", True))
        self.cache_size = max(1, int(getattr(self.args, "recursive_kv_cache_size", 32)))
        self._student_eval_cache = OrderedDict()
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
        for key in ["input_ids", "inputs_embeds", "attention_mask", "image_grid_thw", "image_sizes"]:
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
        output_attentions: bool = False,
        output_hidden_states: bool = True,
    ):
        if not self.enable_kv_cache:
            if enable_grad:
                return model.encode_input(
                    input_data,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            with torch.no_grad():
                return model.encode_input(
                    input_data,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

        # Cache only eval-student calls. Teacher caching is intentionally disabled
        # because cached entries include full hidden-state stacks and can trigger
        # cumulative GPU-memory growth / OOM in long training runs.
        if enable_grad:
            return model.encode_input(
                input_data,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if model_tag.startswith("student_eval") and (not model.training):
            cache = self._student_eval_cache
        else:
            with torch.no_grad():
                return model.encode_input(
                    input_data,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
        key = self._cache_key(model_tag, input_data)
        key = f"{key}|attn={int(output_attentions)}|h={int(output_hidden_states)}"
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        with torch.no_grad():
            output = model.encode_input(
                input_data,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
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

    def _step_embedding(self, step: int, dim: int, device, dtype, learned_step_embeddings=None):
        if learned_step_embeddings is not None:
            step = int(step)
            max_step = learned_step_embeddings.num_embeddings - 1
            step_idx = min(max(step, 0), max_step)
            learned = learned_step_embeddings.weight[step_idx].to(device=device, dtype=dtype)
            return learned.view(1, 1, dim)
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
        learned_step_embeddings=None,
        capture_first_step_attn: bool = False,
    ):
        """
        Build recursive student token states:
            X^{k+1} = X^k + f_theta(X^k + e_k), with first pass X^1 = f_theta(X^0 + e_0)

        Here f_theta is one *full* student forward pass each step.
        """
        _, _, _, hidden_states = base_output
        x_prev = hidden_states[-1]  # (B, S, D), X^0 from encoder output states
        updates = []

        student_eval_cache_ok = not student_model.training
        attention_mask = student_input["attention_mask"]
        final_reps = base_output[0]
        first_step_attention = None
        first_step_image_features = None
        for k in range(k_steps):
            grad_enabled_step = (not student_eval_cache_ok) and (k >= k_steps - self.backprop_steps)
            step_tag = f"student_eval_step_{k+1}" if student_eval_cache_ok else f"student_train_step_{k+1}"

            e_k = self._step_embedding(
                k,
                x_prev.size(-1),
                x_prev.device,
                x_prev.dtype,
            )
            step_input = {
                "inputs_embeds": x_prev + e_k,
                "attention_mask": attention_mask,
            }
            for key in ("position_ids", "cache_position", "image_grid_thw", "image_sizes", "rope_deltas"):
                if key in student_input:
                    step_input[key] = student_input[key]

            step_output = self._encode_with_cache(
                student_model,
                step_tag,
                step_input,
                enable_grad=grad_enabled_step,
                output_attentions=False,
                output_hidden_states=(k < (k_steps - 1)),
            )
            _, step_image_features, step_attention, step_hidden = step_output
            if capture_first_step_attn and k == 0:
                first_step_attention = step_attention
                first_step_image_features = step_image_features
            f_theta_out = step_hidden[-1]
            final_reps = step_output[0]

            # Last recursive pass is used to produce representation only.
            if k == (k_steps - 1):
                continue
            if k == 0:
                x_next = f_theta_out
            else:
                x_next = x_prev + f_theta_out
            update = x_next - x_prev
            if not grad_enabled_step:
                # Keep non-backprop steps off GPU to reduce peak memory on long CLS batches.
                update = update.detach().to("cpu")
            updates.append(update)
            x_prev = x_next

        return updates, final_reps, first_step_attention, first_step_image_features

    def _update_loss_for_side(
        self,
        student_input,
        teacher_input,
        teacher_hidden_states,
        recursive_student_updates,
        student_image_features,
        teacher_image_features,
        student_special_ids,
        teacher_special_ids,
        projectors,
    ):
        projector_t2s = projectors["t2s"] if "t2s" in projectors else None

        k_steps = len(recursive_student_updates)
        if k_steps == 0:
            zero = teacher_hidden_states[0].new_tensor(0.0)
            return zero, zero
        teacher_idx = self._step_indices(len(teacher_hidden_states), k_steps)

        s_text_counts = count_clean_text_tokens(student_input, student_special_ids)
        t_text_counts = count_clean_text_tokens(teacher_input, teacher_special_ids)

        bsz = student_input["input_ids"].size(0)
        total_mean = teacher_hidden_states[0].new_tensor(0.0)
        total_cov = teacher_hidden_states[0].new_tensor(0.0)
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
                if delta_s.device != total_mean.device:
                    delta_s = delta_s.to(total_mean.device, non_blocking=True)
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
                # normalize token vectors before covariance alignment
                z_s_img_norm = F.normalize(z_s_img, dim=-1)
                z_t_img_norm = F.normalize(z_t_img, dim=-1)
                z_s_txt_norm = F.normalize(z_s_txt, dim=-1)
                z_t_txt_norm = F.normalize(z_t_txt, dim=-1)
                cov_s_img = self._covariance(z_s_img_norm)
                cov_t_img = self._covariance(z_t_img_norm)
                cov_s_txt = self._covariance(z_s_txt_norm)
                cov_t_txt = self._covariance(z_t_txt_norm)
                if cov_s_img is not None and cov_t_img is not None:
                    cov_loss = cov_loss + F.mse_loss(cov_s_img, cov_t_img)
                if cov_s_txt is not None and cov_t_txt is not None:
                    cov_loss = cov_loss + F.mse_loss(cov_s_txt, cov_t_txt)

                wk = 1.0 / float(k_steps)
                total_mean = total_mean + wk * mean_loss
                total_cov = total_cov + wk * cov_loss
                denom += 1

            cur_s_img += 1
            cur_t_img += 1

        if denom == 0:
            zero = total_mean.new_tensor(0.0)
            return zero, zero
        return total_mean / denom, total_cov / denom

    def _contrastive_similarity_distill(self, student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps, projectors):
        teacher_q = F.normalize(projectors["t2s"](teacher_qry_reps), dim=-1)
        teacher_p = F.normalize(projectors["t2s"](teacher_pos_reps), dim=-1)

        student_q = F.normalize(student_qry_reps, dim=-1)
        student_p = F.normalize(student_pos_reps, dim=-1)
        s_sim = student_q @ student_p.transpose(0, 1)
        t_sim = teacher_q @ teacher_p.transpose(0, 1)
        return F.l1_loss(s_sim, t_sim)

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher
        projectors = distiller.projectors
        if not hasattr(self, "_step_embedding_mode_logged"):
            self._step_embedding_mode_logged = True
            print("[RecursiveDistillationLoss] using sinusoidal step embeddings.")
            print(
                f"[RecursiveDistillationLoss] recursive_num_steps={self.num_steps}, "
                f"recursive_backprop_steps={self.backprop_steps}"
            )

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
            output_attentions=False,
            output_hidden_states=True,
        )
        teacher_pos_output = self._encode_with_cache(
            teacher_model,
            "teacher_pos",
            teacher_pos_input,
            enable_grad=False,
            output_attentions=False,
            output_hidden_states=True,
        )

        student_eval_cache_ok = not student_model.training
        student_qry_output = self._encode_with_cache(
            student_model,
            "student_eval_qry_base" if student_eval_cache_ok else "student_train_qry_base",
            student_qry_input,
            enable_grad=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        student_pos_output = self._encode_with_cache(
            student_model,
            "student_eval_pos_base" if student_eval_cache_ok else "student_train_pos_base",
            student_pos_input,
            enable_grad=False,
            output_attentions=False,
            output_hidden_states=False,
        )

        teacher_qry_reps, teacher_qry_image_features, _, teacher_qry_hidden_states = teacher_qry_output
        teacher_pos_reps, teacher_pos_image_features, _, teacher_pos_hidden_states = teacher_pos_output
        student_qry_reps, student_qry_image_features, _, _ = student_qry_output
        student_pos_reps, student_pos_image_features, _, _ = student_pos_output

        def _has_vision_feats(feats):
            if feats is None:
                return False
            try:
                return len(feats) > 0
            except TypeError:
                return True

        has_vision = (
            _has_vision_feats(student_qry_image_features)
            and _has_vision_feats(student_pos_image_features)
            and _has_vision_feats(teacher_qry_image_features)
            and _has_vision_feats(teacher_pos_image_features)
        )

        # On text-only / CLS batches, skip recursive token-state unrolling to avoid
        # building large unused graphs. Recursive mean/cov losses are vision-aware.
        # final_student_*_reps are pooled representations from the LAST recursive pass.
        recursive_qry_updates, final_student_qry_reps, _, _ = self._recursive_student_updates(
            student_model,
            student_qry_input,
            student_qry_output,
            self.num_steps,
            capture_first_step_attn=False,
        )
        recursive_pos_updates, final_student_pos_reps, _, _ = self._recursive_student_updates(
            student_model,
            student_pos_input,
            student_pos_output,
            self.num_steps,
            capture_first_step_attn=False,
        )

        # IMPORTANT: contrastive task loss uses reps from the final recursive pass.
        contrastive_student_qry_reps = final_student_qry_reps
        contrastive_student_pos_reps = final_student_pos_reps
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(contrastive_student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(contrastive_student_pos_reps)
        else:
            all_student_qry_reps = contrastive_student_qry_reps
            all_student_pos_reps = contrastive_student_pos_reps

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

        if has_vision:
            mean_qry_loss, cov_qry_loss = self._update_loss_for_side(
                student_qry_input,
                teacher_qry_input,
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
        else:
            mean_loss = contrastive_loss.new_tensor(0.0)
            cov_loss = contrastive_loss.new_tensor(0.0)
        update_loss = self.mean_weight * mean_loss + self.cov_weight * cov_loss

        # Temporarily disable attention alignment; keep a zero scalar for logging compatibility.
        attn_loss = update_loss.new_tensor(0.0)

        # L1 contrastive KD must use representations from the LAST recursive pass.
        student_qry_reps_for_l1 = contrastive_student_qry_reps
        student_pos_reps_for_l1 = contrastive_student_pos_reps
        contrastive_kd_loss = self._contrastive_similarity_distill(
            student_qry_reps_for_l1,
            student_pos_reps_for_l1,
            teacher_qry_reps,
            teacher_pos_reps,
            projectors,
        )

        kd_loss = update_loss + self.contrastive_weight * contrastive_kd_loss
        loss = contrastive_loss + self.kd_loss_weight * kd_loss
        # DDP safety: ensure all trainable projector params participate in graph,
        # even when some branches/weights are disabled for a given batch.
        projector_guard = loss.new_tensor(0.0)
        for p in projectors.parameters():
            if p.requires_grad:
                projector_guard = projector_guard + p.sum() * 0.0
        step_emb_guard = loss.new_tensor(0.0)
        step_emb_module = getattr(distiller, "recursive_step_embeddings", None)
        if step_emb_module is not None:
            for p in step_emb_module.parameters():
                if p.requires_grad:
                    step_emb_guard = step_emb_guard + p.sum() * 0.0
        loss = loss + projector_guard + step_emb_guard

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
