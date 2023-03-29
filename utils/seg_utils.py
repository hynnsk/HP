from typing import Dict, Any
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment
from utils.dist_utils import all_reduce_dict

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(img_tensor, prob_tensor):
    batch_size = list(img_tensor.size())[0]
    img_tensor_cpu = img_tensor.detach().cpu()
    prob_tensor_cpu = prob_tensor.detach().cpu()
    out = []
    for i in range(batch_size):
        out_ = dense_crf(img_tensor_cpu[i], prob_tensor_cpu[i])
        out.append(out_)

    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in out], dim=0)


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.stats = torch.zeros(n_classes + self.extra_clusters, n_classes,
                                           dtype=torch.int64, device="cuda")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1) # 32*320*320
            preds = preds.reshape(-1) # 32*320*320

            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:  # cluster
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)  # row, col
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]

            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:  # linear
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}


def get_metrics(m1: UnsupervisedMetrics, m2: UnsupervisedMetrics) -> Dict[str, Any]:
    metric_dict_1 = m1.compute()
    metric_dict_2 = m2.compute()
    metrics = all_reduce_dict(metric_dict_1, op="mean")
    tmp = all_reduce_dict(metric_dict_2, op="mean")
    metrics.update(tmp)

    return metrics
