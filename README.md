# FastKD

* [Introduction](##Introduction)
* [Features](##Features)
* [Methods Comparison](##Methods-Comparison)
* [Configuration](##Configuration)
* [Training](##Training)
* [Evaluation](##Evaluation)
* [Inference](##Inference)

## Introduction

PyTorch Knowledge Distillation Framework.

## Features

Datasets:
* [ImageNet](https://image-net.org/)

Sample Model:
* Teacher: ResNet50 (from torchvision)
* Student: ResNet18 (from torchvision)

KD Methods:
* [Vanilla KD](https://arxiv.org/abs/1503.02531)
* [TAKD](https://arxiv.org/abs/1902.03393) (Coming Soon)
* [CRD](http://arxiv.org/abs/1910.10699) (Coming Soon)


Features coming soon:
* [Native DDP](https://pytorch.org/docs/stable/notes/ddp.html)
* [Native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)


## Methods Comparison

Coming Soon...


## Configuration 

Create a configuration file in `configs`. Sample configuration for ImageNet dataset can be found [here](configs/defaults.yaml). Then edit the fields you think if it is needed. This configuration file is needed for both training and evaluation scripts.

## Training

```bash
$ python train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Evaluation

```bash
$ python val.py --cfg configs/CONFIG_FILE_NAME.yaml
```

