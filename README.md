# Compact-Global-Descriptor
The Pytorch implementation of "Compact Global Descriptor (CGD) for Neural Networks" (CGD). [PDF]()

### Toy Illustration
<img src="./img/CGD2.png" width="700" height="240" />

<img src="./img/CGD.png" width="700" height="240" />
CGD is a simple yet effective way to capture the correlations between each position and all positions across channels.

### Formulation


### How to use?
Add an attention layer (CGD) right after the first convolution layer in each block.
#### init
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
