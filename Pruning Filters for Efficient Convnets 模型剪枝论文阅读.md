## 参考文献：
[[1](https://arxiv.org/abs/1608.08710)] Li, Hao, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. **"Pruning filters for efficient convnets."** arXiv preprint arXiv:1608.08710 (2016).

--- 
### 摘要
由于计算机计算能力和存储能力的提升，CNNs在不同领域都大放异彩。  
近来有些工作朝着对不同网络层的权重进行剪枝和压缩方向进行，从而近乎不降低准确率的同时，达到快速推理的目的。  
虽然，基于量的权值剪枝(magnitude-based pruning of weights)大大减少了全连接层的参数，但由于剪枝网络的不规则稀疏性，可能不能充分降低卷积层的计算成本。  
作者提出了一种方法：   
&emsp;将那些对于准确率影响不大的整个滤波器进行剪枝。这么做将会大量减少计算量。  
&emsp;  
相比于对某个权重进行剪枝，作者的方法不会导致稀疏稀疏连接模式(sparse connectivity patterns)。因此，它不需要稀疏卷积库的支持，只需要普通的blas库就可以执行剪枝后的卷积操作。  
大量的实验表明，这种方法可以减少高达34%的VGG-16的推理时间、38%的ResNet-110的推理时间，在CIFAR10上测试，并且通过重新训练剪枝后的网络，能够达到和剪枝前的网络相似的准确度。  

###  引言
