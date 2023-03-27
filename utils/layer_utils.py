import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False, is_direct=False):
        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(self.clusters, dim=1)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
