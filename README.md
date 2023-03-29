# Leveraging Hidden Positives for Unsupervised Semantic Segmentation (CVPR 2023)
Hyun Seok Seong</sup>, WonJun Moon</sup>, SuBeen Lee</sup>, Jae-Pil Heo</sup>

This is the official pytorch implementation of "Leveraging Hidden Positives for Unsupervised Semantic Segmentation".

[[Arxiv]()] | [[Paper]()]

----------


## Requirements
Install following packages.
```
- python=3.6.9
- pytorch==1.7.1
- torchvision==0.8.2
- torchaudio==0.7.2
- cudatoolkit=11.0
- pytorch-lightning
- matplotlib>=3.3,<3.4
- tqdm>=4.59,<4.60
- scipy>=1.5,<1.6
- hydra-core
- seaborn
- pydensecrf
```

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


