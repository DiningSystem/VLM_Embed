import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .utils import count_clean_text_tokens, get_hidden_text_vision, get_hidden_text


class EOSAttentionKLLoss(nn.Module):
    def __init__(self, args):
        super(EOSAttentionKLLoss, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        self.loss_fn = nn.CrossEntropyLoss()
        self.projection_space = getattr(self.args, "eos_projection_space", "student")

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

    def _attention_pool_with_eos(self, tokens: torch.Tensor, eos_rep: torch.Tensor):
        if tokens is None or tokens.size(0) == 0:
            return eos_rep

        scale = tokens.size(-1) ** -0.5
        scores = torch.matmul(tokens, eos_rep.unsqueeze(-1)).squeeze(-1) * scale
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=0)
        return pooled

    def _sample_text_vision_pooled(
        self,
        hidden_state: torch.Tensor,
        eos_rep: torch.Tensor,
        num_text_tokens: int,
        attention_mask: torch.Tensor,
        num_vision_tokens: int,
    ):
        if num_vision_tokens > 0:
            text_hidden, vision_hidden = get_hidden_text_vision(
                hidden_state,
                num_text_tokens,
                num_vision_tokens,
                attention_mask,
            )
        else:
            text_hidden = get_hidden_text(hidden_state, num_text_tokens, attention_mask)
            vision_hidden = None

        pooled_text = self._attention_pool_with_eos(text_hidden, eos_rep)
        pooled_vision = self._attention_pool_with_eos(vision_hidden, eos_rep)

        return pooled_text, pooled_vision

    def _build_pooled_reps(
        self,
        reps,
        image_features,
        hidden_states,
        attention_mask,
        num_text_tokens,
    ):
        batch_size = reps.size(0)
        pooled_text_reps = []
        pooled_vision_reps = []

        cur_img_idx = 0
        for i in range(batch_size):
            has_image = (
                image_features is not None
                and cur_img_idx < len(image_features)
                and image_features[cur_img_idx] is not None
            )

            num_vision_tokens = image_features[cur_img_idx].size(0) if has_image else 0

            pooled_text, pooled_vision = self._sample_text_vision_pooled(
                hidden_states[-1][i],
                reps[i],
                num_text_tokens[i].item(),
                attention_mask[i],
                num_vision_tokens,
            )
            pooled_text_reps.append(pooled_text)
            pooled_vision_reps.append(pooled_vision)

            if has_image:
                cur_img_idx += 1

        pooled_text_reps = torch.stack(pooled_text_reps, dim=0)
        pooled_vision_reps = torch.stack(pooled_vision_reps, dim=0)

        return pooled_text_reps, pooled_vision_reps

    def _select_projector(self, direction: str):
        if not hasattr(self.distiller, "projectors") or self.distiller.projectors is None:
            return None

        projectors = self.distiller.projectors
        direction_to_keys = {
            "s2t": ["s2t", "s2t_txt", "s2t_img", "proj_ST"],
            "t2s": ["t2s", "t2s_txt", "t2s_img", "proj_TS", "proj_TI"],
        }
        preferred_keys = direction_to_keys[direction]

        if isinstance(projectors, nn.ModuleDict):
            for key in preferred_keys:
                if key in projectors:
                    return projectors[key]
            return None

        if isinstance(projectors, nn.ModuleList) and len(projectors) > 0:
            return projectors[0]

        return None

    def _align_modal_reps(self, teacher_rep: torch.Tensor, student_rep: torch.Tensor):
        if teacher_rep.size(-1) == student_rep.size(-1):
            return teacher_rep, student_rep

        if self.projection_space == "student":
            projector = self._select_projector("t2s")
            if projector is None:
                raise ValueError(
                    "Requested teacher->student alignment but no t2s projector is available in distiller.projectors."
                )
            teacher_rep = projector(teacher_rep)
            if teacher_rep.size(-1) != student_rep.size(-1):
                raise ValueError(
                    f"Teacher projection did not match student dim ({teacher_rep.size(-1)} vs {student_rep.size(-1)})."
                )
            return teacher_rep, student_rep

        if self.projection_space == "teacher":
            projector = self._select_projector("s2t")
            if projector is None:
                raise ValueError(
                    "Requested student->teacher alignment but no s2t projector is available in distiller.projectors."
                )
            student_rep = projector(student_rep)
            if teacher_rep.size(-1) != student_rep.size(-1):
                raise ValueError(
                    f"Student projection did not match teacher dim ({student_rep.size(-1)} vs {teacher_rep.size(-1)})."
                )
            return teacher_rep, student_rep

        raise ValueError(
            f"Invalid eos_projection_space='{self.projection_space}'. Use 'student' or 'teacher'."
        )

    def _kl_from_cosines(self, teacher_anchor: torch.Tensor, student_swap: torch.Tensor):
        teacher_prob = F.softmax(teacher_anchor / self.distiller.temperature, dim=0)
        student_prob = F.softmax(student_swap / self.distiller.temperature, dim=0)
        return F.kl_div(student_prob.log(), teacher_prob, reduction="batchmean")

    def _pair_kd_loss(
        self,
        teacher_text: torch.Tensor,
        teacher_vision: torch.Tensor,
        student_text: torch.Tensor,
        student_vision: torch.Tensor,
    ):
        aligned_teacher_text, aligned_student_text = self._align_modal_reps(teacher_text, student_text)
        aligned_teacher_vision, aligned_student_vision = self._align_modal_reps(teacher_vision, student_vision)

        teacher_anchor = F.cosine_similarity(aligned_teacher_vision, aligned_teacher_text, dim=-1)

        student_change_vision = F.cosine_similarity(aligned_student_vision, aligned_teacher_text.detach(), dim=-1)
        student_change_text = F.cosine_similarity(aligned_teacher_vision.detach(), aligned_student_text, dim=-1)

        loss_change_vision = self._kl_from_cosines(teacher_anchor.detach(), student_change_vision)
        loss_change_text = self._kl_from_cosines(teacher_anchor.detach(), student_change_text)

        return 0.5 * (loss_change_vision + loss_change_text)

    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher

        if getattr(self, "student_processor", None) is None:
            self.student_processor = distiller.get_student_processor()
        if getattr(self, "teacher_processor", None) is None:
            self.teacher_processor = distiller.get_teacher_processor()

        student_qry_input = input_data["student_inputs"]["qry"]
        student_pos_input = input_data["student_inputs"]["pos"]

        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, _, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, _, teacher_pos_hidden_states = teacher_pos_output

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, _, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, _, student_pos_hidden_states = student_pos_output

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
        contrastive_loss = self.loss_fn(scores / self.distiller.temperature, target)

        student_special_ids = torch.tensor(
            self.student_processor.tokenizer.all_special_ids,
            device=student_qry_input["input_ids"].device,
        )
        teacher_special_ids = torch.tensor(
            self.teacher_processor.tokenizer.all_special_ids,
            device=teacher_qry_input["input_ids"].device,
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)
        num_teacher_text_qry_tokens = count_clean_text_tokens(teacher_qry_input, teacher_special_ids)
        num_teacher_text_pos_tokens = count_clean_text_tokens(teacher_pos_input, teacher_special_ids)

        student_qry_text, student_qry_vision = self._build_pooled_reps(
            student_qry_reps,
            student_qry_image_features,
            student_qry_hidden_states,
            student_qry_input["attention_mask"],
            num_student_text_qry_tokens,
        )
        student_pos_text, student_pos_vision = self._build_pooled_reps(
            student_pos_reps,
            student_pos_image_features,
            student_pos_hidden_states,
            student_pos_input["attention_mask"],
            num_student_text_pos_tokens,
        )

        teacher_qry_text, teacher_qry_vision = self._build_pooled_reps(
            teacher_qry_reps,
            teacher_qry_image_features,
            teacher_qry_hidden_states,
            teacher_qry_input["attention_mask"],
            num_teacher_text_qry_tokens,
        )
        teacher_pos_text, teacher_pos_vision = self._build_pooled_reps(
            teacher_pos_reps,
            teacher_pos_image_features,
            teacher_pos_hidden_states,
            teacher_pos_input["attention_mask"],
            num_teacher_text_pos_tokens,
        )

        qry_kd_loss = self._pair_kd_loss(
            teacher_qry_text,
            teacher_qry_vision,
            student_qry_text,
            student_qry_vision,
        )
        pos_kd_loss = self._pair_kd_loss(
            teacher_pos_text,
            teacher_pos_vision,
            student_pos_text,
            student_pos_vision,
        )

        kd_loss = 0.5 * (qry_kd_loss + pos_kd_loss)
        loss = contrastive_loss + self.kd_loss_weight * kd_loss

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": kd_loss,
        }
