from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size

class StegoLoss(nn.Module):

    def __init__(self,
                 n_classes: int,
                 cfg: dict,
                 corr_weight: float = 1.0):
        super().__init__()

        self.n_classes = n_classes
        self.corr_weight = corr_weight
        self.corr_loss = ContrastiveCorrelationLoss(cfg)
        self.linear_loss = LinearLoss(cfg)

    def forward(self, model_input, model_output, model_pos_output=None, linear_output: torch.Tensor() = None,
                cluster_output: torch.Tensor() = None) \
            -> Tuple[torch.Tensor, Dict[str, float]]:
        img, label = model_input
        # feats, code = model_output
        feats = model_output[0]
        code = model_output[1]

        if self.corr_weight > 0:
            # feats_pos, code_pos = model_pos_output
            feats_pos = model_pos_output[0]
            code_pos = model_pos_output[1]
            corr_loss, corr_loss_dict = self.corr_loss(feats, feats_pos, code, code_pos)
        else:
            corr_loss_dict = {"none": 0}
            corr_loss = torch.tensor(0, device=feats.device)

        linear_loss = self.linear_loss(linear_output, label, self.n_classes)
        cluster_loss = cluster_output[0]
        loss = linear_loss + cluster_loss
        loss_dict = {"loss": loss.item(), "corr": corr_loss.item(), "linear": linear_loss.item(),
                     "cluster": cluster_loss.item()}

        return loss, loss_dict, corr_loss_dict


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg["pointwise"]:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg["zero_clamp"]:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg["stabilize"]:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor,
                orig_feats_pos: torch.Tensor,
                orig_code: torch.Tensor,
                orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg["feature_samples"], self.cfg["feature_samples"], 2]

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg["corr_loss"]["pos_intra_shift"])
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg["corr_loss"]["pos_inter_shift"])

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg["neg_samples"]):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg["corr_loss"]["neg_inter_shift"])
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)

        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (self.cfg["corr_loss"]["pos_intra_weight"] * pos_intra_loss.mean() +
                self.cfg["corr_loss"]["pos_inter_weight"] * pos_inter_loss.mean() +
                self.cfg["corr_loss"]["neg_inter_weight"] * neg_inter_loss.mean(),
                {"self_loss": pos_intra_loss.mean().item(),
                 "knn_loss": pos_inter_loss.mean().item(),
                 "rand_loss": neg_inter_loss.mean().item()}
                )


class LinearLoss(nn.Module):

    def __init__(self, cfg: dict):
        super(LinearLoss, self).__init__()
        self.cfg = cfg
        self.linear_loss = nn.CrossEntropyLoss()

    def forward(self, linear_logits: torch.Tensor, label: torch.Tensor, n_classes: int):
        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < n_classes)

        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
        linear_loss = self.linear_loss(linear_logits[mask], flat_label[mask]).mean()

        return linear_loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, modeloutput_z, modeloutput_s_pr=None, modeloutput_f=None,
                Pool_ag=None, Pool_sp=None, opt=None, lmbd=None, modeloutput_z_mix=None):
        device = (torch.device('cuda')
                  if modeloutput_z.is_cuda
                  else torch.device('cpu'))

        batch_size = modeloutput_z.shape[0]

        spatial_size = opt["model"]["spatial_size"]

        split = int(spatial_size*spatial_size)
        mini_iters = int(batch_size/split)

        negative_mask_one = torch.scatter(torch.ones((split,batch_size), dtype=torch.float16), 1,
                                        torch.arange(split).view(-1,1),0).to(device)
        mask_neglect_base = torch.FloatTensor(split,batch_size).uniform_() < opt["rho"]
        mask_neglect_base = mask_neglect_base.type(torch.float16)
        mask_neglect_base = mask_neglect_base.cuda()

        loss = torch.tensor(0).to(device)
        for mi in range(mini_iters):
            modeloutput_f_one = modeloutput_f[mi*split : (mi+1)*split]
            with torch.cuda.amp.autocast(enabled=True):
                output_cossim_one = torch.matmul(modeloutput_f_one, modeloutput_f.transpose(0, 1))
                Rpoint = torch.matmul(modeloutput_f, Pool_ag.transpose(0, 1))

            Rpoint = torch.max(Rpoint, dim=1).values
            Rpoint_T = Rpoint.unsqueeze(-1).repeat(1, split)
            output_cossim_one_T = output_cossim_one.transpose(0, 1)
            mask_one_T = (Rpoint_T < output_cossim_one_T)
            mask_one_T = torch.tensor(mask_one_T.transpose(0, 1), dtype=torch.float16)
            Rpoint_one = Rpoint[mi*split : (mi+1)*split]
            Rpoint_one = Rpoint_one.unsqueeze(-1).repeat(1, batch_size)
            mask_one = torch.tensor((Rpoint_one < output_cossim_one), dtype=torch.float16)
            mask_one = torch.logical_or(mask_one, mask_one_T).type(torch.float16)
            neglect_mask = torch.logical_or(mask_one, mask_neglect_base).type(torch.float16)
            neglect_negative_mask_one = negative_mask_one * neglect_mask
            mask_one = mask_one * negative_mask_one

            modeloutput_s_pr_one = modeloutput_s_pr[mi*split : (mi+1)*split]
            with torch.cuda.amp.autocast(enabled=True):
                output_cossim_ema_one = torch.matmul(modeloutput_s_pr_one, modeloutput_s_pr.transpose(0, 1))
                Rpoint_ema = torch.matmul(modeloutput_s_pr, Pool_sp.transpose(0, 1))

            Rpoint_ema = torch.max(Rpoint_ema, dim=1).values
            Rpoint_ema_T = Rpoint_ema.unsqueeze(-1).repeat(1, split)
            output_cossim_ema_one_T = output_cossim_ema_one.transpose(0, 1)
            mask_ema_one_T = (Rpoint_ema_T < output_cossim_ema_one_T)
            mask_ema_one_T = torch.tensor(mask_ema_one_T.transpose(0, 1), dtype=torch.float16)
            Rpoint_ema_one = Rpoint_ema[mi*split : (mi+1)*split]
            Rpoint_ema_one = Rpoint_ema_one.unsqueeze(-1).repeat(1, batch_size)
            mask_ema_one = torch.tensor((Rpoint_ema_one < output_cossim_ema_one), dtype=torch.float16)
            mask_ema_one = torch.logical_or(mask_ema_one, mask_ema_one_T).type(torch.float16)
            mask_ema_one = mask_ema_one * negative_mask_one

            modeloutput_z_one = modeloutput_z[mi*split : (mi+1)*split]
            with torch.cuda.amp.autocast(enabled=True):
                anchor_dot_contrast_one = torch.div(
                    torch.matmul(modeloutput_z_one, modeloutput_z.T),
                    self.temperature)

            logits_max_one, _ = torch.max(anchor_dot_contrast_one, dim=1, keepdim=True)
            logits_one = anchor_dot_contrast_one - logits_max_one.detach()
            exp_logits_one = torch.exp(logits_one) * neglect_negative_mask_one
            log_prob_one = logits_one - torch.log(exp_logits_one.sum(1, keepdim=True))

            if opt["loss_version"] == 1:
                nonzero_idx = torch.where(mask_one.sum(1) != 0.)
                mask_one = mask_one[nonzero_idx]
                log_prob_one = log_prob_one[nonzero_idx]
                mask_ema_one = mask_ema_one[nonzero_idx]
                weighted_mask = torch.tensor(mask_one.detach()) + torch.tensor(mask_ema_one).detach()*lmbd
                if opt["reweighting"] == 1:
                    pnm = torch.tensor(torch.sum(weighted_mask, dim=1), dtype=torch.float32)
                    pnm = (pnm / torch.sum(pnm))
                    pnm = pnm / torch.mean(pnm)
                else:
                    pnm = 1
                mean_log_prob_pos_one = (weighted_mask * log_prob_one).sum(1) / (weighted_mask.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one * pnm)

            elif opt["loss_version"] == 2:
                nonzero_idx = torch.where(mask_one.sum(1) != 0.)
                mask_one = mask_one[nonzero_idx]
                nonzero_idx_ema = torch.where(mask_ema_one.sum(1) != 0.)
                mask_ema_one = mask_ema_one[nonzero_idx_ema]
                if opt["reweighting"] == 1:
                    pnm = torch.tensor(torch.sum(mask_one, dim=1), dtype=torch.float32)
                    pnm = (pnm / torch.sum(pnm))
                    pnm = pnm / torch.mean(pnm)
                    pnm_ema = torch.tensor(torch.sum(mask_ema_one, dim=1), dtype=torch.float32)
                    pnm_ema = (pnm_ema / torch.sum(pnm_ema))
                    pnm_ema = pnm_ema / torch.mean(pnm_ema)
                else:
                    pnm = 1
                    pnm_ema=1
                mean_log_prob_pos_one = (mask_one * log_prob_one[nonzero_idx]).sum(1) / (mask_one.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one * pnm)
                mean_log_prob_pos_one_ema = (mask_ema_one * log_prob_one).sum(1) / (mask_ema_one.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one_ema * pnm_ema) * lmbd

            modeloutput_z_mix_one = modeloutput_z_mix[mi * split: (mi + 1) * split]
            with torch.cuda.amp.autocast(enabled=True):
                anchor_dot_contrast_one_lhp = torch.div(
                    torch.matmul(modeloutput_z_mix_one, modeloutput_z_mix.T),
                    self.temperature)

            logits_max_one_lhp, _ = torch.max(anchor_dot_contrast_one_lhp, dim=1, keepdim=True)
            logits_one_lhp = anchor_dot_contrast_one_lhp - logits_max_one_lhp.detach()
            exp_logits_one_lhp = torch.exp(logits_one_lhp) * neglect_negative_mask_one
            log_prob_one_lhp = logits_one_lhp - torch.log(exp_logits_one_lhp.sum(1, keepdim=True))

            if opt["loss_version"]==1:
                log_prob_one_lhp = log_prob_one_lhp[nonzero_idx]
                mean_log_prob_pos_one_lhp = (weighted_mask * log_prob_one_lhp).sum(1) / (weighted_mask.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one_lhp * pnm)
            elif opt["loss_version"]==2:
                mean_log_prob_pos_one = (mask_one * log_prob_one[nonzero_idx]).sum(1) / (mask_one.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one * pnm)
                mean_log_prob_pos_one_ema = (mask_ema_one * log_prob_one[nonzero_idx_ema]).sum(1) / (mask_ema_one.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one_ema * pnm_ema) * lmbd

            negative_mask_one = torch.roll(negative_mask_one, split, dims=1)

        loss = loss / mini_iters / 2

        return loss
