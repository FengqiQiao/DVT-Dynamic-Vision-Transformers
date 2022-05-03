### ImageNet
- Download the ImageNet (ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar) to this folder 
- Then, move and extract the training and validation images to labeled sub-folders, using the following [shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)
- Rename folder from 'imagenet' to 'ImageNet'
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