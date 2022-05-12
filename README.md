# CIFAR100_ResNet18
# 1.数据
<code>torchvision.datasets.CIFAR100</code>导入训练集和测试集，并进行归一化,batch_size为128.
# 2.训练步骤
导入<code>torchvision.models</code>的resnet18作为baseline.   
对baseline,cutout,mixup,cutmix进行训练，详细代码和输出见<code>CIFAR100.ipynb</code>  
