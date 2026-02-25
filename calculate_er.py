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

def get_unpadded_hidden(hidden_state, attention_mask):
    outputs = []
    for hs, mask in zip(hidden_state, attention_mask):
        outputs.append(hs[mask.bool()])
    return outputs

def get_eranks(model, input):
    attention_mask = input['attention_mask'] # [b, seq_len]
    batch_size = attention_mask.size(0)
    output = model.encode_input(input)
    reps, image_features, attentions, hidden_states = output
    last_unpadded_hidden = get_unpadded_hidden(hidden_states[-1], attention_mask)
    image_feature_ers = []
    hidden_state_ers = []
    for i in range(batch_size):
        image_feature_ers.append(compute_effective_rank(image_features[i]).item())
        hidden_state_ers.append(compute_effective_rank(last_unpadded_hidden[i]).item())
    return image_feature_ers, hidden_state_ers

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

    qry_hidden_ers = []
    qry_image_feature_ers = []
    pos_hidden_ers = []
    pos_image_feature_ers = []

    for batch in tqdm(islice(train_dataloader, 1000), 
                      desc="Encoding for Effective Rank",
                      disable=not is_main_process, 
                      total=min(len(train_dataloader), 1000)):
        batch = to_device(batch, training_args.device)
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # qry_output = model.encode_input(batch['qry'])
                image_feature_ers, hidden_state_ers = get_eranks(model, batch['qry'])
                qry_image_feature_ers.extend(image_feature_ers)
                qry_hidden_ers.extend(hidden_state_ers)
            # print_rank(f"Batch {batch_idx}: Qry Effective Rank = {effective_rank.item():.4f}")
        
        with torch.no_grad():
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # pos_output = model.encode_input(batch['pos'])
                image_feature_ers, hidden_state_ers = get_eranks(model, batch['pos'])
                pos_image_feature_ers.extend(image_feature_ers)
                pos_hidden_ers.extend(hidden_state_ers)
            # print_rank(f"Batch {batch_idx}: Pos Effective Rank = {effective_rank.item():.4f}")
    
    qry_hidden_ers_mean = np.mean(qry_hidden_ers)
    qry_image_feature_ers_mean = np.mean(qry_image_feature_ers)
    pos_hidden_ers_mean = np.mean(pos_hidden_ers)
    pos_image_feature_ers_mean = np.mean(pos_image_feature_ers)

    if is_main_process:
        print(f"Qry Hidden Effective Rank: {qry_hidden_ers_mean:.4f}")
        print(f"Qry Image Feature Effective Rank: {qry_image_feature_ers_mean:.4f}")
        print(f"Pos Hidden Effective Rank: {pos_hidden_ers_mean:.4f}")
        print(f"Pos Image Feature Effective Rank: {pos_image_feature_ers_mean:.4f}")
    
        encode_qry_hidden_path = os.path.join(data_args.encode_output_path, f"qry_hidden_ers.json")
        encode_qry_image_feature_path = os.path.join(data_args.encode_output_path, f"qry_image_feature_ers.json")
        encode_pos_hidden_path = os.path.join(data_args.encode_output_path, f"pos_hidden_ers.json")
        encode_pos_image_feature_path = os.path.join(data_args.encode_output_path, f"pos_image_feature_ers.json")

        # Lưu list (để vẽ histogram sau)
        with open(encode_qry_hidden_path, "w", encoding="utf-8") as f:
            json.dump(qry_hidden_ers, f)

        with open(encode_qry_image_feature_path, "w", encoding="utf-8") as f:
            json.dump(qry_image_feature_ers, f)

        with open(encode_pos_hidden_path, "w", encoding="utf-8") as f:
            json.dump(pos_hidden_ers, f)

        with open(encode_pos_image_feature_path, "w", encoding="utf-8") as f:
            json.dump(pos_image_feature_ers, f)

        # Lưu mean riêng (txt)
        with open(os.path.join(data_args.encode_output_path, "er_mean.txt"), "w") as f:
            f.write(f"qry_hidden_er_mean: {qry_hidden_ers_mean}\n")
            f.write(f"qry_image_feature_er_mean: {qry_image_feature_ers_mean}\n")
            f.write(f"pos_hidden_er_mean: {pos_hidden_ers_mean}\n")
            f.write(f"pos_image_feature_er_mean: {pos_image_feature_ers_mean}\n")
    


if __name__ == "__main__":
    main()