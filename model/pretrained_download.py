'''
@article{hamilton2022unsupervised,
  title={Unsupervised Semantic Segmentation by Distilling Feature Correspondences},
  author={Hamilton, Mark and Zhang, Zhoutong and Hariharan, Bharath and Snavely, Noah and Freeman, William T},
  journal={arXiv preprint arXiv:2203.08414},
  year={2022}
}
'''

from os.path import join, exists
import wget
import os

models_dir = join("./", "pretrained_models")
os.makedirs(models_dir, exist_ok=True)
model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/models/models/"
model_names = ["moco_v2_800ep_pretrain.pth.tar",
               "model_epoch_0720_iter_085000.pth",
               "picie.pkl"]

saved_models_dir = join("./", "saved_models")
os.makedirs(saved_models_dir, exist_ok=True)
saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
saved_model_names = ["cityscapes_vit_base_1.ckpt",
                     "cocostuff27_vit_base_5.ckpt",
                     "picie_and_probes.pth",
                     "potsdam_test.ckpt"]

target_files = [join(models_dir, mn) for mn in model_names] + \
               [join(saved_models_dir, mn) for mn in saved_model_names]

target_urls = [model_url_root + mn for mn in model_names] + \
              [saved_model_url_root + mn for mn in saved_model_names]

for target_file, target_url in zip(target_files, target_urls):
    if not exists(target_file):
        print("\nDownloading file from {}".format(target_url))
        wget.download(target_url, target_file)
    else:
        print("\nFound {}, skipping download".format(target_file))

