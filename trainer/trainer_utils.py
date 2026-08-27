"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import json
import re
import time
import urllib.error
import urllib.request
import hashlib
import subprocess
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def reduce_sums(values, device):
    """Sum scalar metric numerators/denominators across DDP ranks."""
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def safe_swanlab_log(swanlab, data, step=None):
    if not swanlab or not is_main_process():
        return swanlab
    try:
        swanlab.log(data, step=step)
        return swanlab
    except Exception as exc:
        Logger(f'SwanLab log failed; training continues without tracking: {exc}')
        return None


def experiment_config(args):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    try:
        config['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        config['git_commit'] = 'unknown'
    manifest = Path(args.data_path).resolve().parent / 'manifest.json'
    if manifest.exists():
        config['data_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return config


class MetricWindow:
    def __init__(self):
        self.values = {}

    def add(self, name, value, weight=1):
        total, count = self.values.get(name, (0.0, 0.0))
        scalar = value.detach().item() if torch.is_tensor(value) else float(value)
        self.values[name] = (total + scalar * weight, count + weight)

    def means(self, device):
        names = sorted(self.values)
        packed = [part for name in names for part in self.values[name]]
        reduced = reduce_sums(packed, device)
        result = {
            name: reduced[index] / max(reduced[index + 1], 1)
            for name, index in zip(names, range(0, len(reduced), 2))
        }
        self.values.clear()
        return result


@torch.no_grad()
def evaluate_causal_lm(model, loader, device, autocast_ctx, max_batches=0):
    if dist.is_initialized():
        dist.barrier()
    result = None
    if is_main_process():
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        was_training = raw_model.training
        raw_model.eval()
        loss_sum = aux_sum = tokens = samples = 0.0
        for index, (input_ids, labels) in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            with autocast_ctx:
                output = raw_model(input_ids, labels=labels)
            valid_tokens = labels[..., 1:].ne(-100).sum().item()
            loss_sum += output.loss.item() * valid_tokens
            aux_sum += output.aux_loss.item() * input_ids.size(0)
            tokens += valid_tokens
            samples += input_ids.size(0)
        logits_loss = loss_sum / max(tokens, 1)
        aux_loss = aux_sum / max(samples, 1)
        result = {'loss': logits_loss + aux_loss, 'logits_loss': logits_loss, 'aux_loss': aux_loss}
        if was_training:
            raw_model.train()
    if dist.is_initialized():
        dist.barrier()
    return result


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        swanlab_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                swanlab_id = getattr(run, 'id', None) if run else None
            else:
                swanlab_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'swanlab_id': swanlab_id,
            'wandb_id': swanlab_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


def parse_reward_score(text):
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"Reward model returned no numeric score: {text!r}")
    return max(min(float(match.group()), 3.0), -3.0)


class SiliconFlowRewardModel:
    def __init__(self, api_key, model="Qwen/Qwen2.5-7B-Instruct"):
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")
        self.api_key = api_key
        self.model = model

    def get_score(self, messages, response):
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是回答质量评估器。根据正确性、相关性、清晰度给候选回答打分。只输出-3到3之间的一个数字，不要解释。"
                },
                {
                    "role": "user",
                    "content": f"对话上下文：\n{conversation}\n\n候选回答：\n{response}"
                }
            ],
            "temperature": 0,
            "max_tokens": 32
        }).encode()
        request = urllib.request.Request(
            "https://api.siliconflow.cn/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        content = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as result:
                    content = json.load(result)["choices"][0]["message"]["content"]
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise
                error = exc
            except urllib.error.URLError as exc:
                error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
        if content is None:
            Logger(f"SiliconFlow reward请求连续失败，使用中性分数0.0: {error}")
            return 0.0
        try:
            return parse_reward_score(content)
        except ValueError:
            Logger(f"SiliconFlow reward格式异常，使用中性分数0.0: {content!r}")
            return 0.0
