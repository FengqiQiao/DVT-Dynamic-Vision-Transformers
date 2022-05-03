# Study on "Dynamic-Vision-Transformers"

## Experiments
### 1. Experiment setup
#### 1.1 System setup
| System     | Version                                                                                                                  |
|------------|--------------------------------------------------------------------------------------------------------------------------|
| OS         | Ubuntu [18.04.6 LTS](https://releases.ubuntu.com/18.04/ubuntu-18.04.6-desktop-amd64.iso)                                 |
| GPU        | NVIDIA GeForce GTX 1080                                                                                                  |
| CUDA       | 10.1 ([V10.1.105](https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run))  |
| GPU_driver | [510.68.02](https://us.download.nvidia.com/XFree86/Linux-x86_64/510.68.02/NVIDIA-Linux-x86_64-510.68.02.run)             |

#### 1.2 Python environment setup (we use Conda package manager)
| Package     | Version                               |
|-------------|---------------------------------------|
| apex        | [0.1](https://github.com/NVIDIA/apex) |
| Python      | 3.7.10                                |
| pytorch     | 1.8.1                                 |
| pyyaml      | 6.0                                   |
| torchvision | 0.9.1                                 |

#### 1.3 Dataset source (Try using [aria2](https://askubuntu.com/questions/214018/how-to-make-wget-faster-or-multithreading) to speed up download)
| Dataset                    | Source                                                                                     |
|----------------------------|--------------------------------------------------------------------------------------------|
| ImageNet (ILSVRC2012)      | [ImageNet](https://image-net.org/download-images.php)                                      |
| CIFER-10 (Python Version)  | [From Alex Krizhevski's website](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)  |
| CIFER-100 (Python Version) | [From Alex Krizhevski's website](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) |

Reference:  
This tech report (Chapter 3) describes the dataset and the methodology followed when collecting it in much greater detail. Please cite it if you intend to use this dataset.  
- [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

### 2. Dataset preprocess
#### 2.1 ImageNet
- Download the ImageNet from 1.3
- Then, move and extract the training and validation images to labeled sub-folders, using the following [shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)
- The ImageNet dataset should be prepared as follows:
```
ImageNet
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── ...
│   ├── ...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   ├── ...
│   ├── ...
```

### 3. Train (Single-Node multi-process distributed training)
#### Environmental Variables and its meaning

| Env Var              | Meaning                                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| CUDA_VISIBLE_DEVICES | [which device the program use](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) |

The following should helpful with setting up CUDA_VISIBLE_DEVICES
```commandline
>>> import torch

>>> torch.cuda.is_available()
True

>>> torch.cuda.device_count()
1

>>> torch.cuda.current_device()
0

>>> torch.cuda.device(0)
<torch.cuda.device object at 0x7fbb33074d90>

>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce GTX 1080'
```

#### Python flags and its meaning

| Flag                          | Meaning                                                                         |
|-------------------------------|---------------------------------------------------------------------------------|
| -m torch.distributed.launch   | spawns up multiple distributed training processes on each of the training nodes |
| --nproc_per_node=NUM_OF_GPUS  | number of processes per node (should be <= number of GPUs)                      |

#### main.py script args and its meaning

| Arg          | Meaning                                                        |
|--------------|----------------------------------------------------------------|
| model        | Name of model to train                                         |
| batch-size   | input batch size for training (default: 64)                    |
| lr           | learning rate (default 0.01)                                   |
| weight-decay | weight decay (default: 0.005 for adamw)                        |
| amp          | use NVIDIA Apex AMP or Native AMP for mixed precision training |
| img-size     | Image patch size (default: None => model default)              |

#### 3.1 Train DVT_T2t_vit_12 on ImageNet 
```commandline
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./Dynamic-Vision-Transformer/main.py PATH_TO_DATASET --model DVT_T2t_vit_12 --batch-size 128 --lr 2e-3 --weight-decay .03 --amp --img-size 224
```

#### 3.2 Train DVT_T2t_vit_14 on ImageNet
```commandline
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main.py PATH_TO_DATASET --model DVT_T2t_vit_14 --batch-size 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224
```
