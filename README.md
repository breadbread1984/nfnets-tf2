# normalizer free networks

this project implements normalizer free networks introduced in paper [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171).

## generate dataset

download cifar10 with the following command

```shell
python3 create_datasets.py
```

## train

train on cifar10 with the following command

```shell
python3 train.py --model=(F0|F1|F2|F3|F4|F5|F6|F7) --batch_size=<batch size>
```
