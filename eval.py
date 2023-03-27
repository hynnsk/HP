from typing import Dict, Tuple
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.backends import cudnn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
import wandb
import os
from tqdm import tqdm
from utils.common_utils import (save_checkpoint, parse, dprint, time_log, compute_param_norm,
                                freeze_bn, zero_grad_bn, RunningAverage, Timer)
from utils.dist_utils import all_reduce_dict
from utils.wandb_utils import set_wandb
from utils.seg_utils import UnsupervisedMetrics, batched_crf, get_metrics
from build import (build_model, build_criterion, build_dataset, build_dataloader, build_optimizer)
from pytorch_lightning.utilities.seed import seed_everything
from torchvision import datasets, transforms
import numpy as np
from torch.optim import Adam, AdamW
from loss import SupConLoss

def run(opt: dict, is_test: bool = False, is_debug: bool = False):
    is_train = (not is_test)
    seed_everything(seed=0)
    scaler = torch.cuda.amp.GradScaler(init_scale=2048, growth_interval=1000, enabled=True)

    # -------------------- Folder Setup (Task-Specific) --------------------------#
    prefix = "{}/{}_{}".format(opt["output_dir"], opt["dataset"]["data_type"], opt["wandb"]["name"])
    opt["full_name"] = prefix

    # -------------------- Distributed Setup --------------------------#
    if (opt["num_gpus"] == 0) or (not torch.cuda.is_available()):
        raise ValueError("Run requires at least 1 GPU.")

    if (opt["num_gpus"] > 1) and (not dist.is_initialized()):
        assert dist.is_available()
        dist.init_process_group(backend="nccl")  # nccl for NVIDIA GPUs
        world_size = int(dist.get_world_size())
        local_rank = int(dist.get_rank())
        torch.cuda.set_device(local_rank)
        print_fn = partial(dprint, local_rank=local_rank)  # only prints when local_rank == 0
        is_distributed = True
    else:
        world_size = 1
        local_rank = 0
        print_fn = print
        is_distributed = False

    cudnn.benchmark = True

    is_master = (local_rank == 0)
    wandb_save_dir = set_wandb(opt, local_rank, force_mode="disabled" if (is_debug or is_test) else None)

    if not wandb_save_dir:
        wandb_save_dir = os.path.join(opt["output_dir"], opt["wandb"]["name"])
    if is_test:
        wandb_save_dir = "/".join(opt["checkpoint"].split("/")[:-1])

    train_dataset = build_dataset(opt["dataset"], mode="train", model_type=opt["model"]["pretrained"]["model_type"])
    train_loader_memory = build_dataloader(train_dataset, opt["dataloader"], shuffle=True)

    # ------------------------ DataLoader ------------------------------#
    if is_train:
        train_dataset = build_dataset(opt["dataset"], mode="train", model_type=opt["model"]["pretrained"]["model_type"])
        train_loader = build_dataloader(train_dataset, opt["dataloader"], shuffle=True)
    else:
        train_loader = None

    val_dataset = build_dataset(opt["dataset"], mode="val", model_type=opt["model"]["pretrained"]["model_type"])
    val_loader = build_dataloader(val_dataset, opt["dataloader"], shuffle=False,
                                  batch_size=world_size*32)

    # -------------------------- Define -------------------------------#
    net_model, linear_model, cluster_model = build_model(opt=opt["model"],
                                                         n_classes=val_dataset.n_classes,
                                                         is_direct=opt["eval"]["is_direct"])

    device = torch.device("cuda", local_rank)
    net_model = net_model.to(device)
    linear_model = linear_model.to(device)
    cluster_model = cluster_model.to(device)

    model = net_model
    model_m = model

    print_fn("Model:")
    print_fn(model_m)


    # --------------------------- Evaluate with Best --------------------------------#
    loading_dir = os.path.join(opt['output_dir'], opt['checkpoint'])
    checkpoint_loaded = torch.load(f"{loading_dir}/ckpt.pth", map_location=device)
    net_model.load_state_dict(checkpoint_loaded['net_model_state_dict'], strict=True)
    linear_model.load_state_dict(checkpoint_loaded['linear_model_state_dict'], strict=True)
    cluster_model.load_state_dict(checkpoint_loaded['cluster_model_state_dict'], strict=True)

    loss_, metrics_ = evaluate(net_model, linear_model, cluster_model, val_loader, device=device,
                                                                            opt=opt, n_classes=train_dataset.n_classes)
    s = time_log()
    s += f" ------------------- before crf ---------------------\n"
    for metric_k, metric_v in metrics_.items():
        s += f"before crf{metric_k} : {metric_v:.2f}\n"
    print_fn(s)


    loss_, metrics_ = evaluate(net_model, linear_model, cluster_model,
        val_loader, device=device, opt=opt, n_classes=train_dataset.n_classes, is_crf=opt["eval"]["is_crf"])

    s = time_log()
    s += f" -------------------after crf ---------------------\n"
    for metric_k, metric_v in metrics_.items():
        s += f"[after crf] {metric_k} : {metric_v:.2f}\n"
    print_fn(s)


def evaluate(net_model: nn.Module,
             linear_model: nn.Module,
             cluster_model: nn.Module,
             eval_loader: DataLoader,
             device: torch.device,
             opt: Dict,
             n_classes: int,
             is_crf: bool = False,
             data_type: str = "",
             ) -> Tuple[float, Dict[str, float]]:

    net_model.eval()

    cluster_metrics = UnsupervisedMetrics(
        "Cluster_", n_classes, opt["eval"]["extra_clusters"], True)
    linear_metrics = UnsupervisedMetrics(
        "Linear_", n_classes, 0, False)

    with torch.no_grad():
        eval_stats = RunningAverage()

        for i, data in enumerate(tqdm(eval_loader)):
            img: torch.Tensor = data['img'].to(device, non_blocking=True)
            label: torch.Tensor = data['label'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                output = net_model(img)
            feats = output[0]
            head_code = output[1]

            head_code = F.interpolate(head_code, label.shape[-2:], mode='bilinear', align_corners=False)

            if is_crf:
                with torch.cuda.amp.autocast(enabled=True):
                    linear_preds = torch.log_softmax(linear_model(head_code), dim=1)

                with torch.cuda.amp.autocast(enabled=True):
                    cluster_loss, cluster_preds = cluster_model(head_code, 2, log_probs=True, is_direct=opt["eval"]["is_direct"])
                linear_preds = batched_crf(img, linear_preds).argmax(1).cuda()
                cluster_preds = batched_crf(img, cluster_preds).argmax(1).cuda()

            else:
                with torch.cuda.amp.autocast(enabled=True):
                    linear_preds = linear_model(head_code).argmax(1)

                with torch.cuda.amp.autocast(enabled=True):
                    cluster_loss, cluster_preds = cluster_model(head_code, None, is_direct=opt["eval"]["is_direct"])
                cluster_preds = cluster_preds.argmax(1)

            linear_metrics.update(linear_preds, label)
            cluster_metrics.update(cluster_preds, label)

            eval_stats.append(cluster_loss)

        eval_metrics = get_metrics(cluster_metrics, linear_metrics)

        return eval_stats.avg, eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="Path to option JSON file.")
    parser.add_argument("--test", action="store_true", help="Test mode, no WandB, highest priority.")
    parser.add_argument("--debug", action="store_true", help="Debug mode, no WandB, second highest priority.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint override")
    parser.add_argument("--data_path", type=str, default=None, help="Data path override")

    parser_args = parser.parse_args()
    parser_opt = parse(parser_args.opt)
    # if parser_args.checkpoint is not None:
    #     parser_opt["checkpoint"] = parser_args.checkpoint
    if parser_args.data_path is not None:
        parser_opt["dataset"]["data_path"] = parser_args.data_path

    run(parser_opt, is_test=parser_args.test, is_debug=parser_args.debug)


if __name__ == "__main__":
    main()
