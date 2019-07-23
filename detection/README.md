Our implementation heavily bases on [mmdetection](https://github.com/open-mmlab/mmdetection). We would like to thank the authors for code sharing.

### How to use?
* Edit `mmdetection/mmdet/apis/train.py` as follow:
```python
def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # ADD. For CGD weight decay settings
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        if 'w0' in key or 'w1' in key or 'w2' in key or 'bias0' in key or 'bias1' in key or 'bias2' in key or \
            'att_' in key:
            params += [{'params':[value], 'weight_decay': 0.00005}]
            print(key)
        else:
            params += [{'params':[value], 'weight_decay': cfg.optimizer.weight_decay}]

    
    optimizer = optim.SGD(params, lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum)

    # build runner
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # ...
```
* Edit `mmdetection/mmdet/models/backbones/resnet.py` as follow:
```python
  # ...
  self.stride = stride
  self.dilation = dilation
  self.with_cp = with_cp
  self.normalize = normalize
  self.attention = AttentionLayer(planes, planes, True, True) # ADD CGD
  # ...
  
  def _inner_forward(x):
  # ...
  out = self.conv1(x)
  out = self.attention(out) # ATTENTION ADDED
  out = self.norm1(out)
  out = self.relu(out)
  # ...
```
* Add `mobilenet.py` to `mmdetection/mmdet/models/backbones/`
* Add `mobilenet_ssd300_voc.py` to `mmdetection/configs/pascal_voc/`
* Add 'mobilenet_ssd300_coco.py` to `mmdetection/configs/`
