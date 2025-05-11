# Gneration
This project is inspired by[inswapper](https://github.com/haofanwang/inswapper/tree/main)

## Showing
<left><img src="https://github.com/NHY9981/Images_Editing/blob/main/generation/data/id0_0000_00000.png" width="49%" height="49%"></left>
<right><img src="https://github.com/NHY9981/Images_Editing/blob/main/generation/data/id0_id1_0000_00000.png" width="49%" height="49%"></right>

## Installation
```bash

pip install -r requirements.txt
```

You have to install ``onnxruntime-gpu`` manually to enable GPU inference, install ``onnxruntime`` by default to use CPU only inference.


## Download Checkpoints

First, you need to download [face swap model](https://huggingface.co/thebiglaskowski/inswapper_128.onnx/tree/main) and save it under `./checkpoints`.

## DATA

we use ``extract.py`` and ``process_list.py ``to get the target frames and source frames.All frames are from  CelebDF-v2-training.

## Inference

``` bash
python swapper_p.py
        --source_img_folder ./source_frames\
        --target_img_folder ./target_frames \
        --output_img_folder ./output
```
## Acknowledgement
Thanks [insightface.ai](https://insightface.ai/) for releasing their powerful face swap model that makes this happen. 



