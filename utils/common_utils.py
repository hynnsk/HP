from datetime import datetime
from typing import Dict
import time
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
import json
import os
from collections import OrderedDict


def save_checkpoint(prefix: str,
                    net_model, net_optimizer,
                    linear_model, linear_optimizer,
                    cluster_model, cluster_optimizer,
                    current_epoch, current_iter,
                    best_value, save_dir: str,
                    best_epoch=None, best_iter=None,
                    *, model_only: bool = False) -> None:
    model_name = f"{save_dir}/{prefix}.pth"

    if isinstance(net_model, DistributedDataParallel):
        net_model = net_model.module
    if isinstance(linear_model, DistributedDataParallel):
        linear_model = linear_model.module
    if isinstance(cluster_model, DistributedDataParallel):
        cluster_model = cluster_model.module

    torch.save(
        {
            'epoch': current_epoch,
            'iter': current_iter,
            'best_epoch': best_epoch if (best_epoch is not None) else current_epoch,
            'best_iter': best_iter if (best_iter is not None) else current_iter,
            'net_model_state_dict': net_model.state_dict(),
            'net_optimizer_state_dict': net_optimizer.state_dict() if (not model_only) else None,
            'linear_model_state_dict': linear_model.state_dict(),
            'linear_optimizer_state_dict': linear_optimizer.state_dict() if (not model_only) else None,
            'cluster_model_state_dict': cluster_model.state_dict(),
            'cluster_optimizer_state_dict': cluster_optimizer.state_dict() if (not model_only) else None,
            'best': best_value,
        }, model_name)


def parse(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        opt = json.load(f, object_pairs_hook=OrderedDict)  # noqa

    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    opt['num_gpus'] = len(opt['gpu_ids'])

    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    print('number of GPUs=' + str(opt['num_gpus']))

    os.makedirs(opt["output_dir"], exist_ok=True)
    with open(opt['output_dir'] + '/option.json', 'w', encoding='utf-8') as f:
        json.dump(opt, f, indent="\t")

    return opt


def dprint(*args, local_rank: int = 0, **kwargs) -> None:
    if local_rank == 0:
        print(*args, **kwargs)


def time_log() -> str:
    a = datetime.now()
    return f"*" * 48 + f"  {a.year:>4}/{a.month:>2}/{a.day:>2} | {a.hour:>2}:{a.minute:>2}:{a.second:>2}\n"


@torch.no_grad()
def compute_param_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.requires_grad]
    if len(parameters) == 0:
        return torch.as_tensor(0., dtype=torch.float32)

    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p, norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def freeze_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def zero_grad_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            for p in m.parameters():
                # p.grad.fill_(0.0)
                p.grad = None


class RunningAverage:
    def __init__(self):
        self._avg = 0.0
        self._count = 0

    def append(self, value: float) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._avg = (value + self._count * self._avg) / (self._count + 1)
        self._count += 1

    @property
    def avg(self) -> float:
        return self._avg

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._avg = 0.0
        self._count = 0


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self) -> Dict[str, float]:
        return {key: value.avg for key, value in self._dict.items()}

    def reset(self) -> None:
        if self._dict is None:
            return
        for k in self._dict.keys():
            self._dict[k].reset()


class Timer:
    def __init__(self):
        self._now = time.process_time()
        # self._now = time.process_time_ns()

    def update(self) -> float:
        current = time.process_time()
        # current = time.process_time_ns()
        duration = current - self._now
        self._now = current
        return duration / 1e6  # ms
