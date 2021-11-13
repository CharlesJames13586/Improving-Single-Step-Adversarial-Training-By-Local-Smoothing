import argparse
from datetime import datetime
import logging
import os
import utils
import numpy as np
import torch
import data
import models
from apex import amp
from torch import nn
from torch.nn import functional as F
import copy
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'pgd_corner', 'fgsm', 'random_corner', 'free', 'none'], help="训练时使用的攻击方式")
    parser.add_argument('--attack_init', default='random', choices=['zero', 'random'])
    parser.add_argument('--attack_iters', default=10, type=int, help='n_iter of pgd for evaluation')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_eval', default=256, type=int, help='batch size for the final eval with pgd rr; 6 GB memory is consumed for 1024 examples with fp32 network')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'cifar10_binary_gs', 'uniform_noise', 'imagenet'], type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--discrete_eps', action='store_true', help="离散的eps")
    parser.add_argument('--epochs', default=30, type=int, help='15 epochs to reach 45% adv acc, 30 epochs to reach the reported clean/adv accs')
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument('--eval_early_stopped_model', action='store_true', help='whether to evaluate the model obtained via early stopping')
    parser.add_argument('--eval_iter_freq', default=200, type=int, help='how often to evaluate test stats')
    parser.add_argument('--fgsm_alpha', default=1.25, type=float, help="FGSM-AT时的扰动系数")
    parser.add_argument('--grad_align_cos_lambda', default=0.0, type=float, help='coefficient of the cosine gradient alignment regularizer')
    parser.add_argument('--grad_input_sum_coeff', type=float, default=0.0, help='增加到loss上的模型相对输入X的的导数和的系数')
    parser.add_argument('--half_prec', action='store_true', help='if enabled, runs everything as half precision')
    parser.add_argument('--lr_max', default=0.2, type=float, help='0.05 in Table 1, 0.2 in Figure 2')
    parser.add_argument('--lr_schedule', default='cyclic', choices=['cyclic', 'piecewise'])
    parser.add_argument('--mast', action='store_true', help="如果为true，则使用MAST")
    parser.add_argument('--minibatch_replay', default=1, type=int, help='minibatch replay as in AT for Free (default=1 is usual training)')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'lenet', 'cnn'], type=str)
    parser.add_argument('--n_eval_every_k_iter', default=256, type=int, help='on how many examples to eval every k iters')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_final_eval', default=-1, type=int, help='on how many examples to do the final evaluation; -1 means on all test examples.')
    parser.add_argument('--n_restarts', default=1, type=int, help='测试时pgd攻击的迭代次数')
    parser.add_argument('--pgd_alpha_train', default=2.0, type=float)
    parser.add_argument('--pgd_train_n_iters', default=10, type=int, help='n_iter of pgd for training (if attack=pgd)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_eval_epoch', default=0, type=int, help="开始评估的epoch")
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay aka l2 regularization')

    return parser.parse_args()


def main():
    args = get_args()

    cur_timestamp = str(datetime.now())[:-3]                                   # 使用毫秒是为了防止可能存在的命名冲突
    cur_timestamp = cur_timestamp.replace(':', '')                             # 文件名带':'的暂时无法在服务器上创建，所以删除了':'
    model_width = {'linear':'', 'cnn':args.n_filters_cnn, 'lenet':'', 'resnet18':''}[args.model]
    model_str = "{}{}".format(args.model, model_width)
    model_name = "{} dataset={} model={} eps={} attack={} m={} attack_init={} fgsm_alpha={} epochs={} pdg={}-{} grad_align_cos_lambda={} lr_max={} seed={}".format(
        cur_timestamp, args.dataset, model_str, args.eps, args.attack, args.minibatch_replay, args.attack_init, args.fgsm_alpha, args.epochs, args.pgd_alpha_train,
        args.pgd_train_n_iters, args.grad_align_cos_lambda, args.lr_max, args.seed
    )

    if not os.path.exists('models'):
        os.makedirs('models')

    logger = utils.configure_logger(model_name, args.debug)
    logging.info(args)

    half_prec = args.half_prec
    n_cls = 2 if 'binary' in args.dataset else 10

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    double_bp = True if args.grad_align_cos_lambda > 0 else False              # 还不知道为什么要double_bp
    n_eval_every_k_iter = args.n_eval_every_k_iter
    args.pgd_alpha = args.eps / 4                                              # 应该是PGD攻击的alpha

    eps = args.eps / 255                                                       # 参数指定的单位是像素值，改为标准化的值
    pgd_alpha = args.pgd_alpha / 255
    pgd_alpha_train = args.pgd_alpha_train / 255

    train_data_augm = False if args.dataset in ['mnist'] else True
    train_batches = data.get_loaders(args.dataset, -1, args.batch_size, train_set=True, shuffle=True, data_augm=train_data_augm)
    train_batches_fast = data.get_loaders(args.dataset, n_eval_every_k_iter, args.batch_size, train_set=True, shuffle=True, data_augm=False)
    test_batches = data.get_loaders(args.dataset, args.n_final_eval, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)
    test_batches_fast = data.get_loaders(args.dataset, n_eval_every_k_iter, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)

    model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn).cuda()
    model.apply(utils.initialize_weights)
    model.train()

    # 设置 optimizer
    if args.model == "resnet18":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
    elif args.model == "cnn":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    else:
        raise ValueError('decide about the right optimizer for the new model')

    if half_prec:
        if double_bp:
            amp.register_float_function(torch, 'batch_norm')
        model, opt = amp.initialize(model, opt, opt_level="O1")                # 这里是“欧1”，不是“零1”

    if args.attack == "fgsm":                                                  # 只在Free-AT中被用到
        delta = torch.zeros(args.batch_size, *data.shapes_dict[args.dataset][1:]).cuda()
        delta.requires_grad = True

    lr_schedule = utils.get_lr_schedule(args.lr_schedule, args.epochs, args.lr_max)
    loss_function = nn.CrossEntropyLoss()

    train_acc_pgd_best, best_state_dict = 0.0, copy.deepcopy(model.state_dict())
    test_acc_pgd_best, best_state_dict_test = 0.0, copy.deepcopy(model.state_dict())
    start_time = time.time()                                                   # 起始时间
    time_train, iteration, best_iteration = 0, 0, 0
    best_iteration_test = 0
    ###
    eps_discrete_list = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255]
    eps_discrete = 0/255
    ###
    for epoch in range(args.epochs+1):
        print("----------{}----------".format(epoch))
        if args.discrete_eps:
            logger.info("eps:{:.0f}".format(eps_discrete * 255))
        train_loss, train_reg, train_acc, train_n, grad_norm_x, avg_delta_l2 = 0, 0, 0, 0, 0, 0
        train_gis = 0
        for i, (X, y) in enumerate(train_batches):
            if i % args.minibatch_replay != 0 and i > 0:  # take new inputs only each `minibatch_replay` iterations
                X, y = X_prev, y_prev                     # Free-AT 
            
            time_start_iter = time.time()
            # epoch=0 runs only for one iteration (to check the training stats at init)
            if epoch == 0 and i > 0:
                break

            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch-1 + (i+1)/len(train_batches))               # epoch减1是因为第0个epoch被跳过了，i+1是为了跳过0，因为0点的学习率是0，没有意义
            opt.param_groups[0].update(lr=lr)

            if args.attack in ['pgd', 'pgd_corner']:                           # pgd_corner是什么，还没在论文里找到
                pgd_rs = True if args.attack_init == "random" else False
                n_eps_warmup_epochs = 5
                n_iterations_max_eps = n_eps_warmup_epochs * data.shapes_dict[args.dataset][0] // args.batch_size
                eps_pgd_train = min(iteration / n_iterations_max_eps * eps, eps) if args.dataset == "svhn" else eps
                delta = utils.attack_pgd_training(model, X, y, eps_pgd_train, pgd_alpha_train, opt, half_prec, args.pgd_train_n_iters, rs=pgd_rs)
                if args.attack == "pgd_corner":
                    delta = eps * torch.sign(delta)                            # 这种pgd_corner和fgsm有什么区别
                    delta = utils.clamp(X+delta, 0, 1) - X
            
            elif args.attack == "fgsm":
                if args.minibatch_replay == 1:
                    if args.attack_init == "zero":
                        delta = torch.zeros_like(X, requires_grad=True)
                    elif args.attack_init == "random":                         # “random”离散方式需要改变
                        delta = utils.get_uniform_delta(X.shape, eps, requires_grad=True, discrete=args.discrete_eps)
                    else:
                        raise ValueError("Wrong args.attack_init")
                
                else:                                                          # Free-AT，重新使用前面的iteration的存在的delta
                    delta.requires_grad = True

                X_adv = utils.clamp(X+delta, 0, 1)
                output = model(X_adv)
                loss = F.cross_entropy(output, y)
                if half_prec:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        grad = torch.autograd.grad(scaled_loss, delta, create_graph=True if double_bp else False)[0]
                        grad /= scaled_loss / loss
                else:
                    grad = torch.autograd.grad(loss, delta, create_graph=True if double_bp else False)[0]

                grad = grad.detach()
                ### 修改eps为动态
                # argmax_delta = eps * torch.sign(grad)
                if args.discrete_eps:
                    argmax_delta = eps_discrete * torch.sign(grad)
                else:
                    argmax_delta = eps * torch.sign(grad)
                ###
                ### 修改eps为MAST形式
                if args.mast:
                    mix_eps = utils.get_uniform_delta(X.shape, eps, requires_grad=False, discrete=True)
                    argmax_delta = mix_eps * torch.sign(grad)

                n_alpha_warmup_epochs = 5
                n_iterations_max_alpha = n_alpha_warmup_epochs * data.shapes_dict[args.dataset][0] // args.batch_size
                fgsm_alpha = min(iteration / n_iterations_max_alpha * args.fgsm_alpha, args.fgsm_alpha) if args.dataset == "svhn" else args.fgsm_alpha
                if args.discrete_eps:
                    delta.data = utils.clamp(delta.data + fgsm_alpha * argmax_delta, -eps_discrete, eps_discrete)
                else:
                    delta.data = utils.clamp(delta.data + fgsm_alpha * argmax_delta, -eps, eps)
                delta.data = utils.clamp(X + delta.data, 0, 1) - X

            elif args.attack == "random_corner":
                pass

            elif args.attack == "none":
                delta = torch.zeros_like(X, requires_grad=True)

            else:
                raise ValueError("Wrong args.attack")

            # extra FP+BP to calculate the gradient to monitor it
            if args.attack in ['none', 'random_corner', 'pgd', 'pgd_corner']:
                grad = utils.get_input_grad(model, X, y, opt, eps, half_prec, delta_init='none', backprop=args.grad_align_cos_lambda != 0.0)

            delta = delta.detach()

            output = model(X + delta)
            loss = loss_function(output, y)

            reg = torch.zeros(1).cuda()[0]
            if args.grad_align_cos_lambda != 0.0:
                grad2 = utils.get_input_grad(model, X, y, opt, eps, half_prec, delta_init="random_uniform", backprop=True)
                grads_nnz_idx = ((grad**2).sum([1,2,3])**0.5 != 0) * ((grad2**2).sum([1,2,3])**0.5 != 0)               # 这一步还没看懂
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = utils.l2_norm_batch(grad1), utils.l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += args.grad_align_cos_lambda * (1.0 - cos.mean())
                

                loss += reg

            ########## G_I_S
            grad_input_sum = torch.zeros(1).cuda()[0]
            if args.grad_input_sum_coeff != 0.0:
                if args.grad_align_cos_lambda == 0.0:
                    grad2 = utils.get_input_grad(model, X, y, opt, eps, half_prec, delta_init="random_uniform", backprop=True)
                grad_sum = (grad2 * grad2).sum() ** 0.5
                # print(grad_sum)
                # print(loss)
                grad_input_sum = args.grad_input_sum_coeff * grad_sum
                loss += grad_input_sum
                # print(loss)
                


            ##########

            if epoch != 0:
                opt.zero_grad()
                utils.backward(loss, opt, half_prec)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
                opt.step()

            time_train += time.time() - time_start_iter
            train_loss += loss.item() * y.size(0)
            train_reg += reg.item() * y.size(0)                                # Returns the value of this tensor as a standard Python number. This only works for tensors with one element.
            train_gis += grad_input_sum.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)                                               # 已经训练的样本个数

            with torch.no_grad():                                              # no grad 为了统计信息
                grad_norm_x += utils.l2_norm_batch(grad).sum().item()
                delta_final = utils.clamp(X + delta, 0, 1) - X
                avg_delta_l2 += ((delta_final ** 2).sum([1, 2, 3]) ** 0.5).sum().item()
            
            # if iteration % args.eval_iter_freq == 0:
            #     print("reg {:.4f}, g_i_s {:.4f}".format(train_reg, train_gis))
            
            if iteration % args.eval_iter_freq == 0 and epoch > args.start_eval_epoch:
                train_loss, train_reg = train_loss / train_n, train_reg / train_n
                train_gis = train_gis / train_n
                train_acc, avg_delta_l2 = train_acc / train_n, avg_delta_l2 / train_n

                model.eval()

                test_acc_clean, _, _ = utils.rob_acc(test_batches_fast, model, eps, pgd_alpha, opt, half_prec, 0, 1)                    
                test_acc_fgsm, test_loss_fgsm, fgsm_deltas = utils.rob_acc(test_batches_fast, model, eps, eps, opt, half_prec, 1, 1, rs=False)      
                test_acc_pgd, test_loss_pgd, pgd_deltas = utils.rob_acc(test_batches_fast, model, eps, pgd_alpha, opt, half_prec, args.attack_iters, args.n_restarts, rs=True)
                cos_fgsm_pgd = utils.avg_cos_np(fgsm_deltas, pgd_deltas)
                train_acc_pgd, _, _ = utils.rob_acc(train_batches_fast, model, eps, pgd_alpha, opt, half_prec, args.attack_iters, args.n_restarts, rs=True)           # 用于 early stopping

                grad_x = utils.get_grad_np(model, test_batches_fast, eps, opt, half_prec, rs=False)
                grad_eta = utils.get_grad_np(model, test_batches_fast, eps, opt, half_prec, rs=True)
                cos_x_eta = utils.avg_cos_np(grad_x, grad_eta)                 # 论文中的 GradAlign

                time_elapsed = time.time() - start_time                        # 消逝的时间
                train_str = "[train] loss {:.3f}, reg {:.3f}, g_i_s {:.3f} acc {:.2%}, acc_pgd-{}-{} {:.2%}".format(train_loss, train_reg, train_gis, train_acc, args.attack_iters, args.n_restarts, train_acc_pgd)
                test_str = "[test] acc_clean {:.2%}, acc_fgsm {:.2%}, acc_pgd-{}-{} {:.2%}, cos_x_eta {:.3}, cos_fgsm_pgd {:.3}".format(test_acc_clean, test_acc_fgsm, args.attack_iters, args.n_restarts, test_acc_pgd, cos_x_eta, cos_fgsm_pgd)
                logger.info("{}-{}: {}  {} ({:.2f}m, {:.2f}m)".format(epoch, iteration, train_str, test_str, time_train/60, time_elapsed/60))

                # 为什么要用训练集的，不用测试集的
                if train_acc_pgd > train_acc_pgd_best:                         # 在训练集上可以检测到灾难性过拟合(catastriphic overfitting)
                    best_state_dict = copy.deepcopy(model.state_dict())
                    train_acc_pgd_best, best_iteration = train_acc_pgd, iteration
                    
                if test_acc_pgd > test_acc_pgd_best:
                    best_state_dict_test = copy.deepcopy(model.state_dict())
                    test_acc_pgd_best, best_iteration_test = test_acc_pgd, iteration


                model.train()
                train_loss, train_reg, train_acc, train_n, grad_norm_x, avg_delta_l2 = 0, 0, 0, 0, 0, 0
                train_gis = 0
            
            iteration += 1
            X_prev, y_prev = X.clone(), y.clone()

        if epoch == args.epochs:
            torch.save({"last": model.state_dict(), "best": best_state_dict, "best_test": best_state_dict_test}, "models/{} epoch={}.pth".format(model_name, epoch))
            # disable global conversion to fp16 from amp.initialize() (https://github.com/NVIDIA/apex/issues/567)
            context_manager = amp.disable_casts() if half_prec else utils.nullcontext()
            with context_manager:
                last_state_dict = copy.deepcopy(model.state_dict())
                half_prec = False  # final eval is always in fp32
                model.load_state_dict(last_state_dict)
                # utils.model_eval(model, half_prec)
                model.eval()
                opt = torch.optim.SGD(model.parameters(), lr=0)

                attack_iters, n_restarts = (50, 10) if not args.debug else (10, 3)
                test_acc_clean, _, _ = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                test_acc_pgd_rr, _, deltas_pgd_rr = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
                logger.info('[last: test on 10k points] acc_clean {:.2%}, pgd_rr {:.2%}'.format(test_acc_clean, test_acc_pgd_rr))

                if args.eval_early_stopped_model:
                    model.load_state_dict(best_state_dict)
                    # utils.model_eval(model, half_prec)
                    model.eval()
                    test_acc_clean, _, _ = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                    test_acc_pgd_rr, _, deltas_pgd_rr = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
                    logger.info('[best: test on 10k points][iter={}] acc_clean {:.2%}, pgd_rr {:.2%}'.format(
                        best_iteration, test_acc_clean, test_acc_pgd_rr))
                    
                    model.load_state_dict(best_state_dict_test)
                    model.eval()
                    test_acc_clean, _, _ = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                    test_acc_pgd_rr, _, deltas_pgd_rr = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
                    logger.info('[best-test: test on 10k points][iter={}] acc_clean {:.2%}, pgd_rr {:.2%}'.format(
                        best_iteration_test, test_acc_clean, test_acc_pgd_rr))
        ###
        if args.discrete_eps:
            # eps_discrete += 1/255
            # if eps_discrete > eps:
            #     eps_discrete = 1/255
            eps_discrete = eps_discrete_list[(epoch) % len(eps_discrete_list)]
        ###

        model.train()
    
    logger.info("Done in {:.2f}m".format((time.time() - start_time) / 60))



if __name__ == "__main__":
    main()