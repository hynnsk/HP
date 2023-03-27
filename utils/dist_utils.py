from typing import Any, Dict, List
from numbers import Number
import torch
import torch.distributed as dist


__all__ = [
    "all_reduce_scalar",
    "all_reduce_tensor",
    "all_reduce_dict",
    "all_gather_tensor",
]


def all_reduce_scalar(value: Number, op: str = "sum") -> Number:
    """All-reduce single scalar value. NOT torch tensor."""
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py
    if dist.is_initialized() and dist.is_available():
        op = op.lower()
        if (op == "sum") or (op == "mean"):
            dist_op = dist.ReduceOp.SUM
        elif op == "min":
            dist_op = dist.ReduceOp.MIN
        elif op == "max":
            dist_op = dist.ReduceOp.MAX
        elif op == "product":
            dist_op = dist.ReduceOp.PRODUCT
        else:
            raise RuntimeError(f"Invalid all_reduce op: {op}")

        backend = dist.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device("cuda")
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device("cpu")
        else:
            raise RuntimeError(f"Unsupported distributed backend: {backend}")

        tensor = torch.tensor(value, device=device, requires_grad=False)
        dist.all_reduce(tensor, op=dist_op)
        if op == "mean":
            tensor /= dist.get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret


def all_reduce_tensor(tensor: torch.Tensor, op="sum", detach: bool = True) -> torch.Tensor:
    if dist.is_initialized() and dist.is_available():
        ret = tensor.clone()
        if detach:
            ret = ret.detach()
        if (op == "sum") or (op == "mean"):
            dist_op = dist.ReduceOp.SUM
        else:
            raise RuntimeError(f"Invalid all_reduce op: {op}")

        dist.all_reduce(ret, op=dist_op)
        if op == "mean":
            ret /= dist.get_world_size()
    else:
        ret = tensor
    return ret


def all_reduce_dict(result: Dict[str, Any], op="sum") -> Dict[str, Any]:
    new_result = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            new_result[k] = all_reduce_tensor(v, op)
        elif isinstance(v, Number):
            new_result[k] = all_reduce_scalar(v, op)
        else:
            raise RuntimeError(f"Dictionary all_reduce should only have either tensor or scalar, got: {type(v)}")
    return new_result


def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    if dist.is_initialized() and dist.is_available():
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        output = [
            tensor if (i == local_rank) else torch.empty_like(tensor) for i in range(world_size)
        ]
        dist.all_gather(output, tensor, async_op=False)
        return output
    else:
        return [tensor]
