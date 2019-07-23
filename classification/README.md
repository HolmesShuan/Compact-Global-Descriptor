Our implementation bases on [pytorch-classification](https://github.com/bearpaw/pytorch-classification). We would like to thank the authors for code sharing.

### How to use?
1. Please refer to the [training recipes](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md) in [pytorch-classification](https://github.com/bearpaw/pytorch-classification) for CIFAR-10 training.
2. We provide the distributed training script for ImageNet training. E.g.,
```shell
python imagenet.py -a resnet50 --data /data/ilsvrc12_torch/ --epochs 101 --schedule 31 61 81 --gamma 0.1 -c checkpoints/imagenet/resnet50 --gpu-id 0,1,2,3,4,5,6,7 --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -j16
python imagenet.py -a mobilenet --data /data/ilsvrc12_torch/ --epochs 91 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/mobilenet-2 --gpu-id 0,1,2,3,4,5,6,7 --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -j16 --wd 4e-5

```
