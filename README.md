# Leveraging Hidden Positives for Unsupervised Semantic Segmentation (CVPR 2023)
Hyun Seok Seong</sup>, WonJun Moon</sup>, SuBeen Lee</sup>, Jae-Pil Heo</sup>

This is the official pytorch implementation of "Leveraging Hidden Positives for Unsupervised Semantic Segmentation".

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/leveraging-hidden-positives-for-unsupervised/unsupervised-semantic-segmentation-on-coco-7)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-coco-7?p=leveraging-hidden-positives-for-unsupervised)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/leveraging-hidden-positives-for-unsupervised/unsupervised-semantic-segmentation-on-potsdam-1)](https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-potsdam-1?p=leveraging-hidden-positives-for-unsupervised)

[[Arxiv](https://arxiv.org/abs/2303.15014)] | [[Paper]()]

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
Change the `pytorch_data_dir` variable in `dataset_download.py` according to your data directory where datasets are stored and run as:
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

## Acknowledgement
This repository is built based on [STEGO](https://github.com/mhamilton723/STEGO) repository.
Thanks for the great work.


