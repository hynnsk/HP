import torch
import torch.nn as nn
import model.dino.vision_transformer as vits

class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):  # cfg["pretrained"]
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg["pretrained"]["dino_patch_size"]
        self.patch_size = patch_size
        self.feat_type = self.cfg["pretrained"]["dino_feat_type"]
        arch = self.cfg["pretrained"]["model_type"]
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        self.n_feats = 384
        self.const = 28

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg["pretrained"]["pretrained_weights"] is not None:
            state_dict = torch.load(cfg["pretrained"]["pretrained_weights"], map_location="cpu")
            state_dict = state_dict["teacher"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(
                cfg["pretrained"]["pretrained_weights"], msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg["pretrained"]["projection_type"]
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

        self.ema_model1 = self.make_clusterer(self.n_feats)
        self.ema_model2 = self.make_nonlinear_clusterer(self.n_feats)

        for param_q, param_k in zip(self.cluster1.parameters(), self.ema_model1.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model1.cuda()
        self.ema_model1.eval()

        for param_q, param_k in zip(self.cluster2.parameters(), self.ema_model2.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model2.cuda()
        self.ema_model2.eval()

        sz = cfg["spatial_size"]

        self.index_mask = torch.zeros((sz*sz, sz*sz), dtype=torch.float16)
        self.divide_num = torch.zeros((sz*sz), dtype=torch.long)
        for _im in range(sz*sz):
            if _im == 0:
                index_set = torch.tensor([_im, _im+1, _im+sz, _im+(sz+1)])
            elif _im==(sz-1):
                index_set = torch.tensor([_im-1, _im, _im+(sz-1), _im+sz])
            elif _im==(sz*sz-sz):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1])
            elif _im==(sz*sz-1):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im])

            elif ((1 <= _im) and (_im <= (sz-2))):
                index_set = torch.tensor([_im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            elif (((sz*sz-sz+1) <= _im) and (_im <= (sz*sz-2))):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1])
            elif (_im % sz == 0):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1, _im+sz, _im+(sz+1)])
            elif ((_im+1) % sz == 0):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im, _im+(sz-1), _im+sz])
            else:
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            self.index_mask[_im][index_set] = 1.
            self.divide_num[_im] = index_set.size(0)

        self.index_mask = self.index_mask.cuda()
        self.divide_num = self.divide_num.unsqueeze(1)
        self.divide_num = self.divide_num.cuda()


    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer_layer3(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    @torch.no_grad()
    def ema_model_update(self, model, ema_model, ema_m):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
            param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

        for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def forward(self, img, n=1, return_class_feat=False, train=False):
        self.model.eval()
        batch_size = img.shape[0]

        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            if train==True:
                attn = attn[:, :, 1:, 1:]
                attn = torch.mean(attn, dim=1)
                attn = attn.type(torch.float32)
                attn_max = torch.quantile(attn, 0.9, dim=2, keepdim=True)
                attn_min = torch.quantile(attn, 0.1, dim=2, keepdim=True)
                attn = torch.max(torch.min(attn, attn_max), attn_min)

                attn = attn.softmax(dim=-1)
                attn = attn*self.const
                attn[attn < torch.mean(attn, dim=2, keepdim=True)] = 0.

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            code_ema = self.ema_model1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
                code_ema += self.ema_model2(self.dropout(image_feat))
        else:
            code = image_feat

        if train==True:
            attn = attn * self.index_mask.unsqueeze(0).repeat(batch_size, 1, 1)
            code_clone = code.clone()
            code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
            code_clone = code_clone.permute(0,2,1)

            code_3x3_all = []
            for bs in range(batch_size):
                code_3x3 = attn[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
                code_3x3 = torch.sum(code_3x3, dim=1)
                code_3x3 = code_3x3 / self.divide_num
                code_3x3_all.append(code_3x3)
            code_3x3_all = torch.stack(code_3x3_all)
            code_3x3_all = code_3x3_all.permute(0,2,1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        if train==True:
            with torch.no_grad():
                self.ema_model_update(self.cluster1, self.ema_model1, self.cfg["ema_m"])
                self.ema_model_update(self.cluster2, self.ema_model2, self.cfg["ema_m"])

        if train==True:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema), self.dropout(code_3x3_all)
            else:
                return image_feat, code, code_ema, code_3x3_all
        else:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema)
            else:
                return image_feat, code, code_ema
