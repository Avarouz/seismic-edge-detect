[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tiny-and-efficient-model-for-the-edge/edge-detection-on-uded)](https://paperswithcode.com/sota/edge-detection-on-uded?p=tiny-and-efficient-model-for-the-edge)

# Tiny and Efficient Model for the Edge Detection Generalization (Paper)

## Overview

<div style="text-align:center"><img src='imgs/teedBanner.png' width=800>
</div>



Tiny and Efficient Edge Detector (TEED) is a light convolutional neural
network with only $58K$ parameters, less than $0.2$% of the 
state-of-the-art models. Training on the [BIPED](https://www.kaggle.com/datasets/xavysp/biped)
dataset takes *less than 30 minutes*, with each epoch requiring 
*less than 5 minutes*. Our proposed model is easy to train
and it quickly converges within very first few epochs, while the 
predicted edge-maps are crisp and of high quality, see image above.
[This paper has been accepted by ICCV 2023-Workshop RCV](https://arxiv.org/abs/2308.06468).

... In construction

    git clone https://github.com/xavysp/TEED.git
    cd TEED

Then,


# EDITS (12/2025)

## Data
    data/
    ├── BIPED/
    │   ├── images/
    │   ├── edges/
    │   ├── train_rgb.lst      # image_path label_path (one per line)
    │   └── test_rgb.lst

## Testing with TEED

Copy and paste your images into data/ folder, and:

    python main.py --is_testing \
    --train_data BIPED \
    --choose_test_data 1 \
    --input_val_dir /path/to/test/data \
    --test_list test_rgb.lst \
    --checkpoint_data checkpoints/BIPED/49_model.pth

## Training with TEED


In console, run:

    python main.py --is_training \
    --train_data BIPED \
    --test_data BIPED \
    --input_dir /path/to/BIPED \
    --input_val_dir /path/to/BIPED \
    --train_list train_rgb.lst \
    --test_list test_rgb.lst \
    --epochs 50 \
    --batch_size 8

Check the configurations of the datasets in dataset.py


## UDED dataset

Here the [link](https://github.com/xavysp/UDED) to access the UDED dataset for edge detection



Please cite our Dataset if you find helpful in your academic/scientific publication,
```
@InProceedings{Soria_2023teed,
    author    = {Soria, Xavier and Li, Yachuan and Rouhani, Mohammad and Sappa, Angel D.},
    title     = {Tiny and Efficient Model for the Edge Detection Generalization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {1364-1373}
}