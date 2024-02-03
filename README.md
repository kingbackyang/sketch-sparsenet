# *Sketch-SparseNet*

This is the official implementation of "***Sketch-SparseNet***: A Novel Sparse-convolution-based Framework for Sketch Recognition"

## Supported Backbone

- [x] ResNet 18/34/50/101
- [x] MnasNet
- [x] Mobilenet
- [x] DenseNet
- [x] Swin Transformer

## Supported Datasets

- [x] QuickDraw-414k
- [x] Tuberlin 
- [x] CIFAR

## Installation

### Prerequisites

The code is built with following libraries:

- Python >= 3.6, <3.8
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.6
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)
- [numba](http://numba.pydata.org/)
- [cv2](https://github.com/opencv/opencv)

## Training

Supported Distributed Training

You can modify the config (e.g. ***configs/swin_image.yaml***) to choose or define your model for training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python train_img_single.py configs/swin_image.yaml --run-dir nbenchmark/swin_sce # 4 gpus
```

```
CUDA_VISIBLE_DEVICES=2,3 torchpack dist-run -np 2 python train_img2.py configs/quickdraw/sd3b1_image_stroke.yaml --run-dir nbenchmark/trans/resnet50_quickdraw_image_stroke_sd3b1_norm/
```

Single GPU Training

```
python train_img_single.py configs/swin_image.yaml --run-dir nbenchmark/swin_sce --distributed False
```

```
python train_img2.py configs/quickdraw/sd3b1_image_stroke.yaml --run-dir nbenchmark/trans/resnet50_quickdraw_image_stroke_sd3b1_norm/ --distributed False
```



# Citation

> 

# Issues

If you have any problems, feel free to reach out to me in the issue.
