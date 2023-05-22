# Leveraging Hidden Positives for Unsupervised Semantic Segmentation (CVPR 2023)
Hyun Seok Seong</sup>, WonJun Moon</sup>, SuBeen Lee</sup>, Jae-Pil Heo</sup>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/leveraging-hidden-positives-for-unsupervised/unsupervised-semantic-segmentation-on-potsdam-1)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-potsdam-1?p=leveraging-hidden-positives-for-unsupervised)
<br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/leveraging-hidden-positives-for-unsupervised/unsupervised-semantic-segmentation-on-coco-7)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-coco-7?p=leveraging-hidden-positives-for-unsupervised)

[[Arxiv](https://arxiv.org/abs/2303.15014)] | [[Paper]()]

## Abstract
> Dramatic demand for manpower to label pixel-level annotations triggered the advent of unsupervised semantic segmentation. Although the recent work employing the vision transformer (ViT) backbone shows exceptional performance, there is still a lack of consideration for task-specific training guidance and local semantic consistency. To tackle these issues, we leverage contrastive learning by excavating hidden positives to learn rich semantic relationships and ensure semantic consistency in local regions. Specifically, we first discover two types of global hidden positives, task-agnostic and task-specific ones for each anchor based on the feature similarities defined by a fixed pre-trained backbone and a segmentation head-in-training, respectively. A gradual increase in the contribution of the latter induces the model to capture task-specific semantic features. In addition, we introduce a gradient propagation strategy to learn semantic consistency between adjacent patches, under the inherent premise that nearby patches are highly likely to possess the same semantics. Specifically, we add the loss propagating to local hidden positives, semantically similar nearby patches, in proportion to the predefined similarity scores. With these training schemes, our proposed method achieves new state-of-the-art (SOTA) results in COCO-stuff, Cityscapes, and Potsdam-3 datasets.
----------


## Requirements
Install following packages.
```
- python=3.6.9
- pytorch
- pytorch-lightning
- matplotlib
- tqdm
- scipy
- hydra-core
- seaborn
- pydensecrf
```

## Prepare datasets
Change the `pytorch_data_dir` variable in `dataset_download.py` according to your data directory where datasets are stored and run:
```
python ./dataset/dataset_download.py
```
Then, extract the zip files.

## Training & Evaluation
You should modify the data path in "<path_to_HP>/json/server/cocostuff.json" according to your dataset path.

```data_path
"dataset": {
        "data_type": "cocostuff27",
        "data_path": "<YOUR_COCOSTUFF_PATH>",
```

To train the model, run the code as below:
```train
python run.py --opt ./json/server/cocostuff.json
```

To evaluate, you should modify the checkpoint path in "<path_to_HP>/json/server/cocostuff_eval.json" according to the saved checkpoint path:
```ckpt_path
"output_dir": "./output/",
"checkpoint": "hp_saved",
```

Then run the evaluation code as below:
```
python eval.py --opt ./json/server/cocostuff_eval.json
```

Note that all of our experiments are tested on single A6000 GPU.

## Acknowledgement
This repository is built based on [STEGO](https://github.com/mhamilton723/STEGO) repository.
Thanks for the great work.

## Licence
Our codes are released under [MIT](https://opensource.org/licenses/MIT) license.

## Citation
If you find this project useful, please consider the following citation:
```
@article{seong2023leveraging,
  title={Leveraging Hidden Positives for Unsupervised Semantic Segmentation},
  author={Seong, Hyun Seok and Moon, WonJun and Lee, SuBeen and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2303.15014},
  year={2023}
}
```
```
@article{seong2023leveraging,
  author    = {Seong, Hyun Seok and Moon, WonJun and Lee, SuBeen and Heo, Jae-Pil},
  title     = {Leveraging Hidden Positives for Unsupervised Semantic Segmentation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
}
```
