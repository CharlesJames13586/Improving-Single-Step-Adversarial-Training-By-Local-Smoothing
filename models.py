import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs

        
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1                                                              # 本项目中 expansion 好像没有用到

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def relu_with_stats(self, preact):
        """ReLU前统计信息"""
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())                # 为什么要abs()
        act = F.relu(preact)
        return act

    def forward(self, x):
        out = self.relu_with_stats(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu_with_stats(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True
        self.in_planes = 64
        self.avg_preact = None
        self.mu = torch.tensor((0.0, 0.0, 0.0,)).view(1, 3, 1, 1)
        self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.mu = self.mu.half()
        
        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n_cls)
        
        layers = [self.normalize, self.conv1, self.layer1[0].bn1]
        self.model_preact_hl1 = nn.Sequential(*layers)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)                                # block中第一层的步长是stride，后续的都是1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride))
            self.in_planes = planes * block.expansion                          # 调整下一层的输入特征层数(其值等于本层的输出特征层的层数乘以block的expansion)
        
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []
        x = self.normalize(x)
        out = self.conv1(x)
        # print("conv1(x) {}".format(out.mean()))
        out = self.layer1(out)
        # print("layer1(x) {}".format(out.mean()))
        out = self.layer2(out)
        # print("layer2(x) {}".format(out.mean()))
        out = self.layer3(out)
        # print("layer3(x) {}".format(out.mean()))
        out = self.layer4(out)
        # print("layer4(x) {}".format(out.mean()))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        avg_preacts_all = []
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            avg_preacts_all += layer.avg_preacts
        self.avg_preact = np.mean(avg_preacts_all)

        return out


def PreActResNet18(n_cls, cuda=True, half_prec=False):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, cuda=cuda, half_prec=half_prec)



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ReluWithStats(nn.Module):
    def __init__(self):
        super(ReluWithStats, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []

    def forward(self, preact):
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())
        act = F.relu(preact)
        return act


class ModuleWithStats(nn.Module):
    def __init__(self):
        super(ModuleWithStats, self).__init__()

    def forward(self, x):
        for layer in self._model:
            if type(layer) == ReluWithStats:
                layer.avg_preacts = []

        out = self._model(x)

        avg_preacts_all = [layer.avg_preacts for layer in self._model if type(layer) == ReluWithStats]
        self.avg_preact = np.mean(avg_preacts_all)
        return out


class CNNBase(ModuleWithStats):
    """ Just needed to provide a generic implementation of calc_distances_hl1(self, X) """
    def __init__(self):
        super(CNNBase, self).__init__()

    def calc_distances_hl1(self, X):
        conv1 = self._model[0].weight
        first_conv_norm_channelwise = conv1.abs().sum((1, 2, 3))  # note: l1 distance is implemented!
        # when w==0 and b==0
        first_conv_norm_channelwise[first_conv_norm_channelwise + self._model[0].bias.abs() < 1e-6] = np.nan
        distances = self.model_preact_hl1(X).abs() / first_conv_norm_channelwise[None, :, None, None]
        distances = distances.view(X.shape[0], -1)

        return distances


class CNN(CNNBase):
    def __init__(self, n_cls, shape_in, n_conv, n_filters):
        super(CNN, self).__init__()
        input_size = shape_in[2]
        conv_blocks = []
        for i_layer in range(n_conv):
            n_in = shape_in[1] if i_layer == 0 else n_filters
            n_out = n_filters
            conv_blocks += [nn.Conv2d(n_in, n_out, 3, stride=1, padding=1), ReluWithStats()]
        h_after_conv, w_after_conv = input_size, input_size
        self._model = nn.Sequential(
            *conv_blocks,
            Flatten(),
            nn.Linear(n_filters*h_after_conv*w_after_conv, n_cls)
        )
        self.model_preact_hl1 = nn.Sequential(self._model[0])


def get_model(model_name, n_cls, half_prec, shapes_dict, n_filters_cnn):
    if model_name == "resnet18":
        model = PreActResNet18(n_cls, half_prec=half_prec)
    elif model_name == "cnn":
        model = CNN(n_cls, shapes_dict, 1, n_filters_cnn)
    else:
        raise ValueError("Wrong Model")

    return model