# Leveraging Hidden Positives for Unsupervised Semantic Segmentation (CVPR 2023)
Hyun Seok Seong</sup>, WonJun Moon</sup>, SuBeen Lee</sup>, Jae-Pil Heo</sup>

This is the official pytorch implementation of "Leveraging Hidden Positives for Unsupervised Semantic Segmentation".

[[Arxiv]()] | [[Paper]()]

----------


## Requirements
We follow the environment of STEGO. Please check their [github](https://github.com/mhamilton723/STEGO/blob/master/environment.yml).

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


