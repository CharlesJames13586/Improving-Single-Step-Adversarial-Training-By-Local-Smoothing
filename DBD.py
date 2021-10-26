# Decision Boundary Disortion
# todo:
# 1. 画出loss的曲面图         ✅
# 2. 曲面图上的标注预测值的信息 🤔

import numpy as np
from numpy.core.function_base import linspace
import torch
import torch.nn as nn
import argparse
import os
import models
import data
import utils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size_eval', default=256, type=int, help='batch size for the final eval with pgd rr; 6 GB memory is consumed for 1024 examples with fp32 network')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'cifar10_binary_gs', 'uniform_noise', 'imagenet'], type=str)
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument("--timestamp", default="2021-10-12", help="模型训练的时间戳，用来选择被测试的模型")
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'lenet', 'cnn'], type=str)
    parser.add_argument('--model_label', default='last', choices=['last', 'best', 'best_test'], type=str)
    parser.add_argument('--n_final_eval', default=1, type=int, help='on how many examples to do the final evaluation; -1 means on all test examples.')

    return parser.parse_args()


def main():
    args = get_args()
    
    # 1.加载模型和数据
    timestamp = args.timestamp
    model_name = ""
    for filename in os.listdir("models/"):
        if timestamp in filename:
            model_name = "models/" + filename
            break
    model_saved = torch.load(model_name)

    n_cls = 2 if 'binary' in args.dataset else 10
    test_batches = data.get_loaders(args.dataset, args.n_final_eval, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)

    model = models.get_model(args.model, n_cls, False, data.shapes_dict[args.dataset], 16).cuda()
    model.load_state_dict(model_saved[args.model_label])

    images, labels = iter(test_batches).next()
    images, labels = images.cuda(), labels.cuda()

    # 2. 生成对抗样本
    j = 0
    eps = args.eps / 255                                                       # 参数指定的单位是像素值
    opt = torch.optim.SGD(model.parameters(), lr=0)
    grad_x = utils.get_grad_np(model, test_batches, eps, opt, False, rs=False)
    grad_dir = torch.Tensor(grad_x[j]).sign().mul(8/255)
    rand_dir = torch.rand_like(grad_dir).sub(0.5).sign().mul(8/255)
    # print(grad_dir)
    # print(rand_dir)
    # print(grad_dir.sum())
    # print(rand_dir.sum())
    cos_x_eta = utils.avg_cos_np(grad_dir.numpy()[np.newaxis,:], rand_dir.numpy()[np.newaxis,:])
    cos_x_eta = utils.avg_cos_np(grad_dir.numpy()[np.newaxis,:], grad_dir.numpy()[np.newaxis,:])
    print(cos_x_eta)
    rx, ry, zs, colors = cal_perturb(model=model, image=images[j], label=labels[j], vec_x=grad_dir, vec_y=rand_dir, range_x=(0,1), range_y=(0,1), grid_size=50)

    plot_perturb_plt(rx, ry, zs, colors)



# 为什么batch_size会影响发画出来图的情况
# 因为网络中有BatchNorm层，所以batch_size会影响结果，如果想画一个光滑的图，要么batch_size设为1，要为设的足够大，保证所有的样例在同一个batch上（当样例不多时）
def cal_perturb(model, image, label, vec_x, vec_y, range_x, range_y, grid_size=8, loss=nn.CrossEntropyLoss(reduction='none'), batch_size=1, cuda=True):
    rx = np.linspace(*range_x, grid_size)
    ry = np.linspace(*range_y, grid_size)

    images = []
    loss_list = []
    pre_list = []

    if cuda:
        image = image.cuda()
        label = label.cuda()
        vec_x = vec_x.cuda()
        vec_y = vec_y.cuda()

    for j in ry:
        for i in rx:
            images.append(image + i*vec_x + j*vec_y)

            if len(images) == batch_size:
                images = torch.stack(images)
                labels = torch.stack([label]*batch_size)
                outputs = model(images)

                _, pres = torch.max(outputs.data, 1)
                pre_list.append(pres.data.cpu().numpy())
                loss_list.append(loss(outputs, labels).data.cpu().numpy())

                images = []

    if len(images) > 0:
        # 处理剩下的不够一个batch的数据
        images = torch.stack(images)
        labels = torch.stack([label]*len(images))
        outputs = model(images)

        _, pres = torch.max(outputs.data, 1)
        pre_list.append(pres.data.cpu().numpy())
        loss_list.append(loss(outputs, labels).data.cpu().numpy())

    pre_list = np.concatenate(pre_list).reshape(len(rx), len(ry))
    loss_list = np.concatenate(loss_list, axis=None).reshape(len(rx), len(ry))

    return rx, ry, loss_list, pre_list
    

def plot_perturb_plt(rx, ry, loss, predict, color_by_loss=True):
    if color_by_loss:
        colors = loss
    else:
        colors = predict
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(rx, ry)
    ax3.plot_surface(X, Y, loss, rstride = 1, cstride = 1,cmap='rainbow', linewidth=0)
    plt.show()






if __name__ == "__main__":
    main()
