# Compact-Global-Descriptor
The Pytorch implementation of "[Compact Global Descriptor (CGD) for Neural Networks](https://github.com/HolmesShuan/Compact-Global-Descriptor/blob/master/img/egpaper_for_review.pdf)" (CGD). [PDF](https://github.com/HolmesShuan/Compact-Global-Descriptor/blob/master/img/egpaper_for_review.pdf)

### Toy llustration:
<img src="./img/CGD2.png" width="640" height="262" />
CGD is a simple yet effective way to capture the correlations between each position and all positions across channels. 
![equation](http://latex.codecogs.com/gif.latex?f) and ![equation](http://latex.codecogs.com/gif.latex? g) correspond to the global avg/max pooling which map features across spatial dimensions into a response vector.

### Formulation:
![equation](http://latex.codecogs.com/gif.latex?\psi(X)&=\text{Tanh}(\text{Softmax}(\text{pool}_{ave}(X))\text{pool}_{ave}(X)^Tw))

![equation](http://latex.codecogs.com/gif.latex?\phi(X)&=\text{Tanh}(\text{Softmax}(\text{pool}_{ave}(X))\text{pool}_{max}(X)^Tw'))

![equation](http://latex.codecogs.com/gif.latex?\text{CGD}(X)&=X(1+\text{Tanh}(\psi(X)\phi(X)^Tw''))) 

See [attention_best.py](https://github.com/HolmesShuan/Compact-Global-Descriptor/blob/master/attention_best.py).

### How to use?
Add an attention layer (CGD) right after the first convolution layer in each block. Set the weight decay of CGD to 4e-5.
#### init:
```python
# __init__(self, in_channels, out_channels, bias=True, nonlinear=True):
self.attention = AttentionLayer(planes, planes, True, True)
```
#### ResNet / MobileNet
```python
out = self.conv1(x)
out = self.attention(out)
out = self.bn1(out)
out = self.relu(out)
```
#### PreResNet
```python
residual = x

out = self.bn1(x)
out = self.relu(out)
out = self.conv1(out)

out = self.attention(out)

out = self.bn2(out)
out = self.relu(out)
out = self.conv2(out)
```
#### SqueezeNet
```python
x = self.squeeze_activation(self.bn(self.attention(self.squeeze(x))))
```
#### WRN
```python
if not self.equalInOut:
    x = self.relu1(self.bn1(x))
else:
    out = self.relu1(self.bn1(x))
out = self.relu2(self.bn2(self.attention(self.conv1(out if self.equalInOut else x))))
```

### Results:
#### ImageNet Acc
<img src="./img/imagenet.png" width="700" height="244" />

#### COCO mAP

<img src="./img/coco.png" width="700" height="202" />


