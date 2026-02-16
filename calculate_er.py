import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
import time

import dataclasses

from sklearn.pipeline import islice

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.single_wrapper import SingleWrapper, SingleCollator, SingleDataset

from transformers import HfArgumentParser, AutoConfig


from src.model.model import MMEBModel
from src.data.dataset.mmeb_dataset import EvalDataset
from src.data.collator.eval_collator import EvalCollator
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor, COLPALI
from torch.nn.utils.rnn import pad_sequence
import shutil 

def delete_pycache(root='.'):
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            if dirname == '__pycache__':
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting: {full_path}")
                try:
                    shutil.rmtree(full_path)
                except:
                    print(">>>>>", "Module not exists", full_path, flush=True)
                    pass
delete_pycache()


def prepare_dataset(data_args, model_args):
    dataset = SingleDataset(data_args, model_args)
    return dataset

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def to_device(obj, device):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    else:
        if hasattr(obj, 'to') and callable(obj.to):
            return obj.to(device)
        return obj

@contextmanager
def time_block(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[Timer] {name}: {elapsed:.4f}s")

def compute_effective_rank(
        hidden_state: torch.Tensor, # [N, D]
        eps: float = 1e-10,
    ) -> torch.Tensor:
    X = hidden_state.float() 
    N = X.size(0)
    s = torch.linalg.svdvals(X) / torch.sqrt(torch.tensor(N))
    eigvals = s * s
    prob = eigvals.clamp(min=eps) / eigvals.sum()
    entropy = -(prob * torch.log(prob)).sum()
    effective_rank = torch.exp(entropy) / N
    return effective_rank.to(dtype=hidden_state.dtype)

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    train_dataset = prepare_dataset(data_args, model_args)

    is_main_process = training_args.local_rank in [-1, 0]

    os.makedirs(data_args.encode_output_path, exist_ok=True)
    
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    # model = MMEBModel.build(model_args)
    # if model_args.load_pretrained_lora:
    #     model.encoder.merge_and_unload()
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    train_dataset = prepare_dataset(data_args, model_args)
    collator = SingleCollator(
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    # random_sample = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        # sampler=random_sample,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
    )

    encode_qry_path = os.path.join(data_args.encode_output_path, f"er_qry")
    encode_tgt_path = os.path.join(data_args.encode_output_path, f"er_tgt")

    qry_er_list = []
    pos_er_list = []
    for batch in tqdm(islice(train_dataloader, 1000), 
                      desc="Encoding for Effective Rank",
                      disable=not is_main_process, 
                      total=min(len(train_dataloader), 1000)):
        batch = to_device(batch, training_args.device)
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                qry_reps = model(qry=batch['qry'])["qry_reps"]
                effective_rank = compute_effective_rank(qry_reps)
                qry_er_list.append(effective_rank.item())
            # print_rank(f"Batch {batch_idx}: Qry Effective Rank = {effective_rank.item():.4f}")
        
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                pos_reps = model(tgt=batch['pos'])["tgt_reps"]
                effective_rank = compute_effective_rank(pos_reps)
                pos_er_list.append(effective_rank.item())
            # print_rank(f"Batch {batch_idx}: Pos Effective Rank = {effective_rank.item():.4f}")
    
    qry_er_mean = float(np.mean(qry_er_list))
    pos_er_mean = float(np.mean(pos_er_list))

    if is_main_process:
        print(f"[ER] qry mean = {qry_er_mean:.6f}")
        print(f"[ER] pos mean = {pos_er_mean:.6f}")

        # Lưu list (để vẽ histogram sau)
        with open(encode_qry_path + ".json", "w", encoding="utf-8") as f:
            json.dump(qry_er_list, f)

        with open(encode_tgt_path + ".json", "w", encoding="utf-8") as f:
            json.dump(pos_er_list, f)

        # Lưu mean riêng (txt)
        with open(os.path.join(data_args.encode_output_path, "er_mean.txt"), "w") as f:
            f.write(f"qry_er_mean: {qry_er_mean}\n")
            f.write(f"pos_er_mean: {pos_er_mean}\n")
    


if __name__ == "__main__":
    main()