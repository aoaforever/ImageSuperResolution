以下翻译至[Pytorch官方文档](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training)  
## 目录
  * [1.模型架构](#模型架构)  
  * [2.辅助函数](#辅助函数)
  * [3.定义数据集和数据加载器](#定义数据集和数据加载器)
  * [4.后训练静态量化](#后训练静态量化)
  * [5.量化感知训练](#量化感知训练)  
  * [从量化加速](#从量化加速)
  * [结论](#结论)
 ---
本教程展示了如何进行训练后的静态量化，并演示了两种更高级的技术——每通道量化和量化感知训练——以进一步提高模型的准确性。  
注意，量化目前只支持cpu，所以在本教程中我们不会使用gpu / CUDA。  
在本教程结束时，您将看到PyTorch中的量化是如何在提高速度的同时显著减小模型大小的。  
此外，您将看到如何轻松地应用这里展示的一些高级量化技术，从而使您的量化模型比其他模型获得更少的精度打击。  
警告:我们使用了大量来自其他PyTorch repos的样板代码，例如，定义MobileNetV2模型架构，定义数据加载器，等等。  
我们当然鼓励你去阅读它;但是如果你想要得到量子化特征，你可以直接跳到“[4](#后训练静态量化)”。训练后静态量化”部分。我们将从必要的导入开始:

```python
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)
```
<span id="模型架构"><span>
### 1.模型架构
我们首先定义了MobileNetV2模型架构，并对其进行了一些显著的修改，以实现量化:  
  * 将加法替换为nn.quantized.FloatFunctional  
  * 在网络的开头和结尾插入QuantStub和DeQuantStub  
  * 将ReLU6替换为ReLU  
代码源自[这里](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenet.py)
```python
from torch.quantization import QuantStub, DeQuantStub

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.quant(x)

        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)
                        
```
<span id="辅助函数"><span>
### 2.辅助函数 
我们接下来定义几个辅助函数来帮助进行模型评估。这些大多来自[这里](https://github.com/pytorch/examples/blob/master/imagenet/main.py)。
```python
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
```
<span id="定义数据集和数据加载器"><span>
### 3.定义数据集和数据加载器
作为最后一个主要的设置步骤，我们为训练和测试集定义数据加载器。  
#### ImageNet Data
要使用整个ImageNet数据集运行本教程中的代码，首先按照ImageNet Data中的说明下载ImageNet。将下载的文件解压到' data_path '文件夹中。  
下载完数据后，我们将在下面展示定义用于读取数据的数据加载器的函数。这些函数大多来自[这里](https://github.com/pytorch/vision/blob/main/references/detection/train.py)。
```python
def prepare_data_loaders(data_path):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
           data_path, split="train",
         transforms.Compose([
                   transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize,
               ]))
    dataset_test = torchvision.datasets.ImageNet(
          data_path, split="val",
              transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  normalize,
              ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test
```
接下来，我们将加载预先训练过的MobileNetV2模型。我们在这里提供从torchvision下载数据的[URL](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenet.py#L9)。

```python
data_path = '~/.data/imagenet'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

# Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
# while also improving numerical accuracy. While this can be used with any model, this is
# especially common with quantized models.

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)
```
最后，为了获得“基线”精度，让我们看看融合模块的非量化模型的精度.  
```python
num_eval_batches = 1000

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)
```
整个模型在50,000张图像的eval数据集上获得了71.9%的准确率。  
这将是我们比较的基线。接下来，让我们尝试不同的量化方法。
  
<span id="后训练静态量化"><span>
### 4.后训练静态量化Post-training static quantization  
`后训练静态量化`不仅包括**将权值从float转换为int**，就像在动态量化中那样，**还包括执行额外的步骤**  
  * 首先通过网络输入一批数据，并计算不同激活的结果分布(具体地说，**这是通过在记录数据的不同点插入观察者模块来实现的**)。  
  * 然后使用这些分布来确定在推断时应该如何量化不同的激活(一种简单的技术是简单地将整个激活范围划分为256个级别，但我们支持更复杂的方法)。  
重要的是，这个额外的步骤允许我们在操作之间传递量化的值，而不是在每个操作之间将这些值转换为浮点数，然后再转换为整数，从而大大提高了速度。
```python
num_calibration_batches = 32

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Fuse Conv, bn and relu
myModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.quantization.default_qconfig
print(myModel.qconfig)
torch.quantization.prepare(myModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

# Calibrate with the training set
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

print("Size of model after quantization")
print_size_of_model(myModel)

top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
```

对于这个量化模型，我们在eval数据集上看到的准确率为56.7%。这是因为我们使用了一个简单的最小/最大观测器来确定量化参数。  
尽管如此，我们还是将模型的大小减少到了3.6 MB以下，几乎减少了4倍。

此外，我们可以通过使用不同的量化配置显著提高精度。我们对x86架构的量化推荐配置重复相同的练习。该配置执行以下操作:  
  * 对每个channel设置不同的量化权重
  * 使用直方图观察者收集激活的直方图，然后以最佳方式选择量化参数。
```python
per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)
```
仅仅改变这种量化配置方法就可以将精度提高到67.3%以上!  
尽管如此，这仍比上述71.9%的基线水平低4%。**让我们尝试量化感知训练。**
 
<span id="量化感知训练"><span>
### 5.量化感知训练Quantization-aware training   
量化感知训练(QAT)是一种通常能获得最高精度的量化方法。  
使用QAT，在向前和向后的训练过程中，所有权重和激活都是“假量化”的:也就是说，浮点值被四入以模拟int8值，但所有计算仍然是用浮点数完成的。  
因此，训练过程中的所有权重调整都是在“意识到”模型最终将被量化这一事实的情况下进行的;  
因此，在量化之后，这种方法通常会比`动态量化`或`训练后的静态量化`产生更高的精度。  


实际执行QAT的整个工作流与之前非常相似:
  * 我们可以像以前一样使用相同的模型:量化感知训练不需要额外的准备。  
  * 我们需要使用qconfig来指定**在权重和激活之后插入哪种伪量化**，而不是指定观察者。  
我们首先定义一个训练函数:  
```python
def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return
```
我们像以前一样融合模块:  
```python
qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
```
最后，准备`prepare_qat`进行“伪量化”，为量化感知训练准备模型。
```PYTHON
torch.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)
```
要训练一个精度高的量化模型，需要在推理时进行精确的数值建模。因此，对于量化感知训练，我们将训练循环修改为:
    * 在训练结束时，将批norm转换为使用运行均值和方差，以更好地匹配推理数值。  
    * 我们还冻结量化器参数(scale和zero-point)并微调权重。  
```PYTHON
num_train_batches = 20

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
for nepoch in range(8):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))
```
  
量化感知训练在整个imagenet数据集上的准确率超过71.5%，接近于浮点精度的71.9%。  
更多关于量化感知训练的信息:  
  * QAT是一个后培训量化技术的超集，允许进行更多调试。 例如，我们可以分析模型的准确性是否受到权重或激活量化的限制。  
  * 我们还可以用浮点来模拟量化模型的准确性，因为我们使用假量化来模拟实际量化算法的数值。 
  * 我们也可以很容易地模拟训练后量化。  
  
### 从量化加速  
最后，让我们确认一下我们上面提到的事情:我们的量化模型真的能更快地执行推断吗?让我们测试:  
```PYTHON
def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)
```
在MacBook pro笔记本电脑上运行这个程序，未量化模型的运行时间为61毫秒，而量化模型的运行时间仅为20毫秒，这说明量化模型比浮点型号的速度要快2-4倍。   

###  结论
在本教程中，我们展示了两种量化方法——训练后静态量化和量化感知训练——描述了它们的“幕后工作”以及如何在PyTorch中使用它们。  
                           
[返回顶部](#目录)




