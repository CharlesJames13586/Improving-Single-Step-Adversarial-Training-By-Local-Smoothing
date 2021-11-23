import logging
import os
from torch import nn
from torch.nn import functional as F
import torch
import math
import numpy as np
from apex import amp
from contextlib import contextmanager
import copy


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d/ %H:%M:%S',
    level=logging.DEBUG
)

def configure_logger(model_name, debug):
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger()
    logger.handlers = []  # remove the default logger

    # add a new logger for stdout
    formatter = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if not debug:
        if not os.path.exists('logs'):
            os.makedirs('logs')
        # add a new logger to a log file
        logger.addHandler(logging.FileHandler('logs/{}.log'.format(model_name)))                   # 含有符号 ":" 的会出错
        # logger.addHandler(logging.FileHandler("logs/2.log"))

    return logger


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        n = module.in_features
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


# 学习率的调整,返回的是一个函数
def get_lr_schedule(lr_schedule_type, n_epochs, lr_max):
    if lr_schedule_type == "cyclic":
        lr_schedule = lambda t: np.interp([t], [0, n_epochs * 2 // 5, n_epochs], [0, lr_max, 0])[0]          # 折线(折线的三个点为(0,0), (n_epochs * 2 // 5, lr_max), (n_epochs, 0))在x=t时的y值
    elif lr_schedule_type == "piecewise":                                      # 分段衰减学习率
        def lr_schedule(t):
            if t / n_epochs < 0.5:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 10
            else:
                return lr_max / 100
    else:
        raise ValueError("Wrong lr_schedule_type, here only support 'cyclic' or 'piecewise")

    return lr_schedule


# 求下界l和上届u之前的X
def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    
    return torch.max(torch.min(X, u), l)


# PGD AT
def attack_pgd_training(model, X, y, eps, alpha, opt, half_prec, attack_iters, rs=True, early_stopping=False):
    delta = torch.zeros_like(X).cuda()
    if rs:                                                                     # 随机初始化起始图片
        delta.uniform_(-eps, eps)
    
    delta.requires_grad = True

    for _ in range(attack_iters):
        output = model(clamp(X+delta, 0, 1))
        loss = F.cross_entropy(output, y)

        if half_prec:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
                delta.grad.mul_(loss.item() / scaled_loss.item())
        else:
            loss.backward()
        grad = delta.grad.detach()

        if early_stopping:
            idx_update = output.max(1)[1] == y                                 # early_stopping只继续攻击还预测正确的样本
        else:
            idx_update = torch.ones(y.shape, dtype=torch.bool)
        
        grad_sign = torch.sign(grad)
        delta.data[idx_update] = (delta + alpha * grad_sign)[idx_update]
        delta.data = clamp(X+delta.data, 0, 1) - X
        delta.data = clamp(delta.data, -eps, eps)
        delta.grad.zero_()

    return delta.detach()


def get_uniform_delta(shape, eps, requires_grad=True, discrete=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    if discrete:
        tensor_discrete(delta)
    delta.requires_grad = requires_grad
    return delta

# 获取输入X在delta_init状态下的对于model的梯度
def get_input_grad(model, X, y, opt, eps, half_prec, delta_init="none", backprop=False):
    if delta_init == "none":
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == "random_uniform":
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == "random_corner":
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else: 
        raise ValueError("Wrong delta init")

    output = model(X + delta)
    loss = F.cross_entropy(output, y)

    if half_prec:
        with amp.scale_loss(loss, opt) as scaled_loss:
            grad = torch.autograd.grad(scaled_loss, delta, create_graph=True if backprop else False)[0]
            grad /= scaled_loss / loss
    else:
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]

    if not backprop:
        grad, delta = grad.detach(), delta.detach()

    return grad


# 返回参数 v 
def l2_norm_batch(v):
    norms = (v**2).sum([1,2,3]) ** 0.5
    return norms


def backward(loss, opt, half_prec):
    if half_prec:
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


# attack_iters == 0 时， 返回 0，函数没有对抗攻击效果
# attack_iters == 1 且 alpha == eps 且 rs == False 时，函数是FGSM攻击
def attack_pgd(model, X, y, eps, alpha, opt, half_prec, attack_iters, n_restarts, rs=True, verbose=False, linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    # verbose: bool 为True时输出运行信息
    # linf_proj: bool 为True时扰动的范围为正无穷范式
    # l2_proj: bool 为True时扰动范围为l2 norm，不能与linf_proj同时为True
    # l2_grad_update: bool 应和l2_proj保持一样的值
    if n_restarts > 1 and not rs:
        raise ValueError("no random step and n_restarts > 1!") 
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)
        
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            if half_prec:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                    delta.grad.mul_(loss.item() / scaled_loss.item())
            
            else:
                loss.backward()
            
            grad = delta.grad.detach()

            # l2_grad_update 是扰动在 l2 norm 内的更新方式
            if not l2_grad_update:
                delta.data = delta + alpha * torch.sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1,2,3], keepdim=True)**0.5
            
            delta.data = clamp(X+delta.data, 0, 1, cuda) - X

            if linf_proj:                                                      # 扰动在正无穷范式
                delta.data = clamp(delta.data, -eps, eps, cuda)
            elif l2_proj:                                                      # 扰动在 l2 norm 内
                delta_norms = (delta.data**2).sum([1,2,3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)

            delta.grad.zero_()
        
        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction="none")
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)

            if verbose:
                print("Restart #{}: best loss{:.3f".format(i_restart, max_loss.mean()))

    max_delta = clamp(X+max_delta, 0, 1, cuda) - X

    return max_delta
        
# 建议离散的eps为1像素
# 目前仅支持linf的离散攻击
# 离散的pgd有终止条件
def attack_pgd_discrete(model, X, y, eps, alpha, opt, half_prec, attack_iters, n_restarts, rs=True, verbose=False, linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    if n_restarts > 1 and not rs:
        raise ValueError("no random step and n_restarts > 1!") 
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            # delta.uniform_(-eps, eps).mul_(255).round_().div_(255)             # 随机初始化的离散化
            delta.uniform_(-eps, eps)
            tensor_discrete(delta)
        
        delta.requires_grad = True
        old_grad = torch.zeros_like(delta)
        for i_iter in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
            if half_prec:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                    delta.grad.mul_(loss.item() / scaled_loss.item())
            
            else:
                loss.backward()
            
            grad = delta.grad.detach()
            if i_iter > 0:
                # if torch.sum(torch.add(torch.sign(grad), torch.sign(old_grad))) == 0:
                if torch.sum(torch.add(torch.sign(grad), torch.sign(old_grad))).abs() < 1:
                    print("提前停止",end='\r')
                    break
            old_grad = copy.deepcopy(grad)
            # l2_grad_update 是扰动在 l2 norm 内的更新方式
            if not l2_grad_update:
                if abs(alpha*255 - round(alpha*255)) > 0.000001:
                    print("alpha不是离散值")
                delta.data = delta + alpha * torch.sign(grad)

            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1,2,3], keepdim=True)**0.5
            
            delta.data = clamp(X+delta.data, 0, 1, cuda) - X

            if linf_proj:                                                      # 扰动在正无穷范式
                delta.data = clamp(delta.data, -eps, eps, cuda)
            elif l2_proj:                                                      # 扰动在 l2 norm 内
                delta_norms = (delta.data**2).sum([1,2,3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)

            delta.grad.zero_()
        
        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction="none")
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)

            if verbose:
                print("Restart #{}: best loss{:.3f".format(i_restart, max_loss.mean()))

    max_delta = clamp(X+max_delta, 0, 1, cuda) - X

    if torch.sub(max_delta, max_delta.mul(255).round().div(255)).abs().sum() > 0.001:
        print("max_delta不是离散值")
        # max_delta.mul_(255).round_().div_(255)
        tensor_discrete(max_delta)

    return max_delta


# 计算在pgd攻击下的正确率
def rob_acc(batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=True, linf_proj=True, l2_grad_update=False, corner=False, verbose=False, cuda=True):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    progress = True if len(batches) >= 20 else False
    print()
    for i, (X, y) in enumerate(batches):
        if progress:
            print("{:.2%} [{}/{}]".format(i/len(batches), i, len(batches)), end='\r')
        if cuda:
            X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=rs, verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda)
        
        

        if corner:
            pgd_delta = clamp(X+eps*torch.sign(pgd_delta), 0, 1, cuda) - X
        
        pgd_delta_proj = clamp(X + eps*torch.sign(pgd_delta), 0, 1, cuda) - X  # needed just for investigation

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
        
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())
        pgd_delta_proj_list.append(pgd_delta_proj.cpu().numpy())

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)

    return robust_acc, avg_loss, pgd_delta_np


def rob_acc_discrete(batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=True, linf_proj=True, l2_grad_update=False, corner=False, verbose=False, cuda=True):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    progress = True if len(batches) >= 9 else False
    print()
    for i, (X, y) in enumerate(batches):
        if progress:
            print("{:.2%} [{}/{}]".format(i/len(batches), i, len(batches)), end='\r')
        if cuda:
            X, y = X.cuda(), y.cuda()

        # 离散的delta
        pgd_delta = attack_pgd_discrete(model, X, y, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=rs, verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda)
        
        if corner:
            pgd_delta = clamp(X+eps*torch.sign(pgd_delta), 0, 1, cuda) - X
        
        pgd_delta_proj = clamp(X + eps*torch.sign(pgd_delta), 0, 1, cuda) - X  # needed just for investigation

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
        
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())
        pgd_delta_proj_list.append(pgd_delta_proj.cpu().numpy())

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)

    return robust_acc, avg_loss, pgd_delta_np


# 求矩阵 v1 和 v2 的余弦相似度
def avg_cos_np(v1, v2):
    norms1 = np.sum(v1**2, (1,2,3), keepdims=True) ** 0.5
    norms2 = np.sum(v2**2, (1,2,3), keepdims=True) ** 0.5
    # print(norms1.shape)
    # print(norms2.shape)
    cos_vals = np.sum((v1/norms1) * (v2/norms2), (1,2,3))                      # 这里应该加个括号吧，虽然看起来这样写好像是对的
    cos_vals[np.isnan(cos_vals)] = 1.0                                         # 防止nans (0/0)
    cos_vals[np.isinf(cos_vals)] = 1.0                                         # 防止+inf或-inf (x/0, -x/0)
    avg_cos = np.mean(cos_vals)

    return avg_cos


# 求 delta 的梯度
def get_grad_np(model, batches, eps, opt, half_prec, rs=False, cross_entropy=True):
    # rs: random start

    grad_list = []
    for i, (X, y) in enumerate(batches):
        X, y = X.cuda(), y.cuda()
        
        if rs:
            delta = get_uniform_delta(X.shape, eps, requires_grad=False)
        else:
            delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True
        logits = model(clamp(X+delta, 0, 1))                                 

        if cross_entropy:
            loss = F.cross_entropy(logits, y)
        else:
            y_onehot = torch.zeros([len(y), 10]).long().cuda()
            y_onehot.scatter(1, y[:, None], 1)
            preds_correct_class = (logits * y_onehot.float()).sum(1, keepdim=True)
            margin = preds_correct_class - logits
            margin += y_onehot.float()
            margin = margin.min(1, keepdim=True)[0]
            loss = F.relu(1 - margin).mean()

        if half_prec:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
                delta.grad.mul_((loss / scaled_loss).item())
        else:
            loss.backward()

        grad = delta.grad.detach().cpu()
        grad_list.append(grad.numpy())
        delta.grad.zero_()
    
    grads = np.vstack(grad_list)
    return grads

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

# 将tensor按照图片像素离散化
def tensor_discrete(tensor):
    tensor.mul_(255).round_().div_(255)


def load_model(timestamp):
    for filename in os.listdir("models/"):
        if timestamp in filename:
            model_name = "models/" + filename
            print(model_name)
            break
    model_saved = torch.load(model_name)
    return model_saved