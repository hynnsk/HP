from typing import Dict, Tuple
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    cudnn.benchmark = True

    world_size=1
    local_rank = 0
    wandb_save_dir = set_wandb(opt, local_rank, force_mode="disabled" if (is_debug or is_test) else None)

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
                                  batch_size=32)

    # -------------------------- Define -------------------------------#
    net_model, linear_model, cluster_model = build_model(opt=opt["model"],
                                                         n_classes=val_dataset.n_classes,
                                                         is_direct=opt["eval"]["is_direct"])

    criterion = build_criterion(n_classes=val_dataset.n_classes,
                                opt=opt["loss"])

    device = torch.device("cuda", 0)
    net_model = net_model.to(device)
    linear_model = linear_model.to(device)
    cluster_model = cluster_model.to(device)

    project_head = nn.Linear(opt['model']['dim'], opt['model']['dim'])
    project_head.cuda()
    head_optimizer = Adam(project_head.parameters(), lr=opt["optimizer"]["net"]["lr"])

    criterion = criterion.to(device)
    supcon_criterion = SupConLoss(temperature=opt["tau"]).to(device)
    pd = nn.PairwiseDistance()

    model = net_model
    model_m = model

    print("Model:")
    print(model_m)

    # ------------------- Optimizer  -----------------------#
    if is_train:
        net_optimizer, linear_probe_optimizer, cluster_probe_optimizer = build_optimizer(
            main_params=model_m.parameters(),
            linear_params=linear_model.parameters(),
            cluster_params=cluster_model.parameters(),
            opt=opt["optimizer"],
            model_type=opt["wandb"]["name"])
    else:
        net_optimizer, linear_probe_optimizer, cluster_probe_optimizer = None, None, None

    start_epoch, current_iter = 0, 0
    best_metric, best_epoch, best_iter = 0, 0, 0

    num_accum = 1

    timer = Timer()

    if opt["model"]["pretrained"]["model_type"] == "vit_small":
        feat_dim = 384
    else:
        feat_dim = 768

    # ---------------------------- memory ---------------------------- #
    with torch.no_grad():
        Pool_ag = torch.zeros((opt["model"]["pool_size"], feat_dim), dtype=torch.float16).cuda()
        Pool_sp = torch.zeros((opt["model"]["pool_size"], opt["model"]["dim"]), dtype=torch.float16).cuda()
        Pool_iter = iter(train_loader_memory)

        for _iter in range(len(train_loader_memory)):
            data = next(Pool_iter)
            img: torch.Tensor = data['img'].to(device, non_blocking=True)

            if _iter >= opt["model"]["pool_size"] / opt["dataloader"]["batch_size"]:
                break
            img = img.cuda()
            with torch.cuda.amp.autocast(enabled=True):
                model_output = net_model(img)

                modeloutput_f = model_output[0].clone().detach()
                modeloutput_f = modeloutput_f.view(modeloutput_f.size(0), modeloutput_f.size(1), -1)

                modeloutput_s_pr = model_output[2].clone().detach()
                modeloutput_s_pr = modeloutput_s_pr.view(modeloutput_s_pr.size(0), modeloutput_s_pr.size(1), -1)

            for _iter2 in range(modeloutput_f.size(0)):
                randidx = np.random.randint(0, model_output[0].size(-1) * model_output[0].size(-2))
                Pool_ag[_iter * opt["dataloader"]["batch_size"] + _iter2] = modeloutput_f[_iter2][:,randidx]

            for _iter2 in range(modeloutput_s_pr.size(0)):
                randidx = np.random.randint(0, model_output[2].size(-1) * model_output[2].size(-2))
                Pool_sp[_iter * opt["dataloader"]["batch_size"] + _iter2] = modeloutput_s_pr[_iter2][:,randidx]

            if _iter % 10 == 0:
                print ("Filling Pool Memory [{} / {}]".format((_iter+1)*opt["dataloader"]["batch_size"], opt["model"]["pool_size"]))

        Pool_ag = F.normalize(Pool_ag, dim=1)
        Pool_sp = F.normalize(Pool_sp, dim=1)

    # --------------------------- Train --------------------------------#
    assert is_train
    max_epoch = opt["train"]["epoch"]
    print_freq = opt["train"]["print_freq"]
    valid_freq = opt["train"]["valid_freq"]
    grad_norm = opt["train"]["grad_norm"]
    freeze_encoder_bn = opt["train"]["freeze_encoder_bn"]
    freeze_all_bn = opt["train"]["freeze_all_bn"]

    best_valid_metrics = dict(Cluster_mIoU=0, Cluster_Accuracy=0, Linear_mIoU=0, Linear_Accuracy=0)
    train_stats = RunningAverage()

    for current_epoch in range(start_epoch, max_epoch):
        print(f"-------- [{current_epoch}/{max_epoch} (iters: {current_iter})]--------")

        g_norm = torch.zeros(1, dtype=torch.float32, device=device)

        net_model.train()
        linear_model.train()
        cluster_model.train()
        project_head.train()

        train_stats.reset()
        _ = timer.update()

        maxiter = len(train_loader) * opt["train"]["epoch"]

        for i, data in enumerate(train_loader):
            trainingiter = current_epoch*len(train_loader) + i
            if trainingiter <= opt["model"]["warmup"]:
                lmbd = 0
            else:
                lmbd = (trainingiter - opt["model"]["warmup"]) / (maxiter - opt["model"]["warmup"])

            # newly initialize
            if i % 100 == 0 and i!= 0:
                with torch.no_grad():
                    Pool_sp = torch.zeros((opt["model"]["pool_size"], opt["model"]["dim"]), dtype=torch.float16).cuda()
                    for _iter, data in enumerate(train_loader_memory):
                        if _iter >= opt["model"]["pool_size"] / opt["dataloader"]["batch_size"]:
                            break
                        img_net: torch.Tensor = data['img'].to(device, non_blocking=True)

                        with torch.cuda.amp.autocast(enabled=True):
                            model_output = net_model(img_net)

                            modeloutput_s_pr = model_output[2].clone().detach()
                            modeloutput_s_pr = modeloutput_s_pr.view(modeloutput_s_pr.size(0), modeloutput_s_pr.size(1), -1)

                        for _iter2 in range(modeloutput_s_pr.size(0)):
                            randidx = np.random.randint(0, model_output[2].size(-1) * model_output[2].size(-2))
                            Pool_sp[_iter * opt["dataloader"]["batch_size"] + _iter2] = modeloutput_s_pr[_iter2][:, randidx]

                        if _iter == 0:
                            print("Filling Pool Memory [{} / {}]".format(
                                (_iter + 1) * opt["dataloader"]["batch_size"], opt["model"]["pool_size"]))

                    Pool_sp = F.normalize(Pool_sp, dim=1)

            img: torch.Tensor = data['img'].to(device, non_blocking=True)
            label: torch.Tensor = data['label'].to(device, non_blocking=True)

            img_aug = data['img_aug'].to(device, non_blocking=True)

            data_time = timer.update()

            if freeze_encoder_bn:
                freeze_bn(model_m.model)
            if 0 < freeze_all_bn <= current_epoch:
                freeze_bn(net_model)

            batch_size = img.shape[0]
            net_optimizer.zero_grad(set_to_none=True)
            linear_probe_optimizer.zero_grad(set_to_none=True)
            cluster_probe_optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)

            model_input = (img, label)

            with torch.cuda.amp.autocast(enabled=True):
                model_output = net_model(img, train=True)
                model_output_aug = net_model(img_aug)

            modeloutput_f = model_output[0].clone().detach().permute(0, 2, 3, 1).reshape(-1, feat_dim)
            modeloutput_f = F.normalize(modeloutput_f, dim=1)

            modeloutput_s = model_output[1].permute(0, 2, 3, 1).reshape(-1, opt["model"]["dim"])

            modeloutput_s_aug = model_output_aug[1].permute(0, 2, 3, 1).reshape(-1, opt["model"]["dim"])

            with torch.cuda.amp.autocast(enabled=True):
                modeloutput_z = project_head(modeloutput_s)
                modeloutput_z_aug = project_head(modeloutput_s_aug)
            modeloutput_z = F.normalize(modeloutput_z, dim=1)
            modeloutput_z_aug = F.normalize(modeloutput_z_aug, dim=1)

            loss_consistency = torch.mean(pd(modeloutput_z, modeloutput_z_aug))

            modeloutput_s_mix = model_output[3].permute(0, 2, 3, 1).reshape(-1, opt["model"]["dim"])
            with torch.cuda.amp.autocast(enabled=True):
                modeloutput_z_mix = project_head(modeloutput_s_mix)
            modeloutput_z_mix = F.normalize(modeloutput_z_mix, dim=1)

            modeloutput_s_pr = model_output[2].permute(0, 2, 3, 1).reshape(-1, opt["model"]["dim"])
            modeloutput_s_pr = F.normalize(modeloutput_s_pr, dim=1)

            loss_supcon = supcon_criterion(modeloutput_z, modeloutput_s_pr=modeloutput_s_pr, modeloutput_f=modeloutput_f,
                                   Pool_ag=Pool_ag, Pool_sp=Pool_sp,
                                   opt=opt, lmbd=lmbd, modeloutput_z_mix=modeloutput_z_mix)


            detached_code = torch.clone(model_output[1].detach())
            with torch.cuda.amp.autocast(enabled=True):
                linear_output = linear_model(detached_code)
                cluster_output = cluster_model(detached_code, None, is_direct=False)

                loss, loss_dict, corr_dict = criterion(model_input=model_input,
                                                       model_output=model_output,
                                                       linear_output=linear_output,
                                                       cluster_output=cluster_output
                                                       )

                loss = loss + loss_supcon + loss_consistency*opt["alpha"]
                # loss = loss / num_accum


            forward_time = timer.update()

            scaler.scale(loss).backward()

            if freeze_encoder_bn:
                zero_grad_bn(model_m)
            if 0 < freeze_all_bn <= current_epoch:
                zero_grad_bn(net_model)

            scaler.unscale_(net_optimizer)

            g_norm = nn.utils.clip_grad_norm_(net_model.parameters(), grad_norm)
            scaler.step(net_optimizer)

            scaler.step(linear_probe_optimizer)
            scaler.step(cluster_probe_optimizer)
            scaler.step(head_optimizer)

            scaler.update()

            current_iter += 1

            backward_time = timer.update()

            loss_dict = all_reduce_dict(loss_dict, op="mean")
            train_stats.append(loss_dict["loss"])

            if i % print_freq == 0:
                lrs = [int(pg["lr"] * 1e8) / 1e8 for pg in net_optimizer.param_groups]
                p_norm = compute_param_norm(net_model.parameters())
                s = time_log()
                s += f"epoch: {current_epoch}, iters: {current_iter} " \
                     f"({i} / {len(train_loader)} -th batch of loader)\n"
                s += f"loss(now/avg): {loss_dict['loss']:.6f}/{train_stats.avg:.6f}\n"
                if len(loss_dict) > 2:
                    for loss_k, loss_v in loss_dict.items():
                        if loss_k != "loss":
                            s += f"-- {loss_k}(now): {loss_v:.6f}\n"
                            if loss_k == "corr":
                                for k, v in corr_dict.items():
                                    s += f"  -- {k}(now): {v:.6f}\n"
                s += f"time(data/fwd/bwd): {data_time:.3f}/{forward_time:.3f}/{backward_time:.3f}\n"
                s += f"LR: {lrs}\n"
                s += f"batch_size x world_size x num_accum: " \
                     f"{batch_size} x {world_size} x {num_accum} = {batch_size * world_size * num_accum}\n"
                s += f"norm(param/grad): {p_norm.item():.3f}/{g_norm.item():.3f}"
                print(s)

            # --------------------------- Valid --------------------------------#
            if ((i + 1) % valid_freq == 0) or ((i + 1) == len(train_loader)):
                _ = timer.update()
                valid_loss, valid_metrics = evaluate(net_model, linear_model,
                                                    cluster_model, val_loader,
                                                     device=device, opt=opt, n_classes=val_dataset.n_classes)

                s = time_log()
                s += f"[VAL] -------- [{current_epoch}/{max_epoch} (iters: {current_iter})]--------\n"
                s += f"[VAL] epoch: {current_epoch}, iters: {current_iter}\n"
                s += f"[VAL] loss: {valid_loss:.6f}\n"

                metric = "All"
                prev_best_metric = best_metric
                if best_metric <= (valid_metrics["Cluster_mIoU"] + valid_metrics["Cluster_Accuracy"] + valid_metrics["Linear_mIoU"] + valid_metrics["Linear_Accuracy"]):
                    best_metric = (valid_metrics["Cluster_mIoU"] + valid_metrics["Cluster_Accuracy"] + valid_metrics["Linear_mIoU"] + valid_metrics["Linear_Accuracy"])
                    best_epoch = current_epoch
                    best_iter = current_iter
                    s += f"[VAL] -------- updated ({metric})! {prev_best_metric:.6f} -> {best_metric:.6f}\n"

                    save_checkpoint(
                        "ckpt", net_model, net_optimizer,
                        linear_model, linear_probe_optimizer,
                        cluster_model, cluster_probe_optimizer,
                        current_epoch, current_iter, best_metric, wandb_save_dir, model_only=True)
                    print ("SAVED CHECKPOINT")

                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {best_valid_metrics[metric_k]:.6f} -> {metric_v:.6f}\n"
                    best_valid_metrics.update(valid_metrics)
                else:
                    now_metric = valid_metrics["Cluster_mIoU"] + valid_metrics["Cluster_Accuracy"] + valid_metrics["Linear_mIoU"] + valid_metrics["Linear_Accuracy"]
                    s += f"[VAL] -------- not updated ({metric})." \
                         f" (now) {now_metric:.6f} vs (best) {prev_best_metric:.6f}\n"
                    s += f"[VAL] previous best was at {best_epoch} epoch, {best_iter} iters\n"
                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {metric_v:.6f} vs {best_valid_metrics[metric_k]:.6f}\n"

                print(s)

                net_model.train()
                linear_model.train()
                cluster_model.train()
                train_stats.reset()

            _ = timer.update()

    checkpoint_loaded = torch.load(f"{wandb_save_dir}/ckpt.pth", map_location=device)
    net_model.load_state_dict(checkpoint_loaded['net_model_state_dict'], strict=True)
    linear_model.load_state_dict(checkpoint_loaded['linear_model_state_dict'], strict=True)
    cluster_model.load_state_dict(checkpoint_loaded['cluster_model_state_dict'], strict=True)
    loss_out, metrics_out = evaluate(net_model, linear_model,
        cluster_model, val_loader, device=device, opt=opt, n_classes=train_dataset.n_classes)
    s = time_log()
    for metric_k, metric_v in metrics_out.items():
        s += f"[before CRF] {metric_k} : {metric_v:.2f}\n"
    print(s)

    checkpoint_loaded = torch.load(f"{wandb_save_dir}/ckpt.pth", map_location=device)
    net_model.load_state_dict(checkpoint_loaded['net_model_state_dict'], strict=True)
    linear_model.load_state_dict(checkpoint_loaded['linear_model_state_dict'], strict=True)
    cluster_model.load_state_dict(checkpoint_loaded['cluster_model_state_dict'], strict=True)
    loss_out, metrics_out = evaluate(net_model, linear_model, cluster_model,
        val_loader, device=device, opt=opt, n_classes=train_dataset.n_classes, is_crf=opt["eval"]["is_crf"])
    s = time_log()
    for metric_k, metric_v in metrics_out.items():
        s += f"[after CRF] {metric_k} : {metric_v:.2f}\n"
    print(s)

    wandb.finish()
    print(f"-------- Train Finished --------")


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
    if parser_args.checkpoint is not None:
        parser_opt["checkpoint"] = parser_args.checkpoint
    if parser_args.data_path is not None:
        parser_opt["dataset"]["data_path"] = parser_args.data_path

    run(parser_opt, is_test=parser_args.test, is_debug=parser_args.debug)


if __name__ == "__main__":
    main()
