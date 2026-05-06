import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.eval_collator import EvalCollator
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.utils import print_rank


@dataclass
class FlickrEvalArguments:
    dataset_name: str = "nlphuji/flickr_1k_test_image_text_retrieval"
    dataset_split: str = "test"
    image_instruction: str = "<|image_1|> Find an image caption describing the given image."
    text_instruction: str = ""


class FlickrImageDataset(Dataset):
    def __init__(self, hf_rows, image_instruction: str):
        self.rows = hf_rows
        self.image_instruction = image_instruction

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return self.image_instruction, row["image"]


class FlickrTextDataset(Dataset):
    def __init__(self, hf_rows, text_instruction: str):
        self.pairs = []
        for row in hf_rows:
            filename = row["filename"]
            for caption in row["caption"]:
                self.pairs.append((text_instruction + caption, filename))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text, _ = self.pairs[idx]
        return text, None


def _batch_to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _compute_recall(similarity: torch.Tensor, query_ids, candidate_ids):
    sorted_indices = torch.argsort(similarity, dim=1, descending=True)
    recall = {1: 0, 5: 0, 10: 0}
    for i, qid in enumerate(query_ids):
        top_ids = [candidate_ids[j.item()] for j in sorted_indices[i, :10]]
        for k in recall:
            if qid in top_ids[:k]:
                recall[k] += 1

    n = len(query_ids)
    for k in recall:
        recall[k] = recall[k] / n
    return recall


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, FlickrEvalArguments))
    model_args, data_args, training_args, flickr_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.encode_output_path, exist_ok=True)

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, "model_backbone", model_backbone)
        setattr(training_args, "model_backbone", model_backbone)
    print_rank(f"model_backbone: {model_args.model_backbone}")

    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    rows = load_dataset(flickr_args.dataset_name, split=flickr_args.dataset_split)

    image_dataset = FlickrImageDataset(rows, image_instruction=flickr_args.image_instruction)
    text_dataset = FlickrTextDataset(rows, text_instruction=flickr_args.text_instruction)

    image_names = [row["filename"] for row in rows]
    text_names = [name for row in rows for name in [row["filename"]] * len(row["caption"])]

    collator = EvalCollator(data_args=data_args, model_args=model_args, processor=processor)

    image_loader = DataLoader(
        image_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    text_loader = DataLoader(
        text_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    use_autocast = training_args.device.startswith("cuda") and torch.cuda.is_available()

    image_embs = []
    with torch.no_grad():
        for batch in tqdm(image_loader, desc="Encode Flickr images"):
            batch = _batch_to_device(batch, training_args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                output = model(qry=batch)
            image_embs.append(output["qry_reps"].detach().float().cpu())
    image_embs = torch.cat(image_embs, dim=0)

    text_embs = []
    with torch.no_grad():
        for batch in tqdm(text_loader, desc="Encode Flickr captions"):
            batch = _batch_to_device(batch, training_args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                output = model(qry=batch)
            text_embs.append(output["qry_reps"].detach().float().cpu())
    text_embs = torch.cat(text_embs, dim=0)

    if model_args.normalize:
        image_embs = torch.nn.functional.normalize(image_embs, p=2, dim=-1)
        text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=-1)

    i2t_recall = _compute_recall(image_embs @ text_embs.T, image_names, text_names)
    t2i_recall = _compute_recall(text_embs @ image_embs.T, text_names, image_names)

    metrics = {
        "i2t_R@1": i2t_recall[1],
        "i2t_R@5": i2t_recall[5],
        "i2t_R@10": i2t_recall[10],
        "t2i_R@1": t2i_recall[1],
        "t2i_R@5": t2i_recall[5],
        "t2i_R@10": t2i_recall[10],
        "mean_R@1": (i2t_recall[1] + t2i_recall[1]) / 2,
        "mean_R@5": (i2t_recall[5] + t2i_recall[5]) / 2,
        "mean_R@10": (i2t_recall[10] + t2i_recall[10]) / 2,
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
