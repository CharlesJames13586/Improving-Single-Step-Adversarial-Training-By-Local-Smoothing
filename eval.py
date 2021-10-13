# todo:
# 1. 对训练模型的鲁棒性测试

import argparse
import torch
import os
import utils
import data
import models


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--attack_iters', default=50, type=int, help='n_iter of pgd for evaluation')
    parser.add_argument('--batch_size_eval', default=256, type=int, help='batch size for the final eval with pgd rr; 6 GB memory is consumed for 1024 examples with fp32 network')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'cifar10_binary_gs', 'uniform_noise', 'imagenet'], type=str)
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument('--half_prec', action='store_true', help='if enabled, runs everything as half precision')
    parser.add_argument("--timestamp", default="2021-09-22 000000", help="模型训练的时间戳，用来选择被测试的模型")
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'lenet', 'cnn'], type=str)
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_final_eval', default=-1, type=int, help='on how many examples to do the final evaluation; -1 means on all test examples.')
    # parser.add_argument('--n_restarts', default=10, type=int, help='测试时pgd攻击的迭代次数')

    return parser.parse_args()

    
def main():
    args = get_args()

    timestamp = args.timestamp
    model_name = ""
    for filename in os.listdir("models/"):
        if timestamp in filename:
            model_name = "models/" + filename
            break
    model_saved = torch.load(model_name)

    half_prec = args.half_prec
    args.pgd_alpha = args.eps / 4                                              # 应该是PGD攻击的alpha
    eps = args.eps / 255                                                       # 参数指定的单位是像素值，改为标准化的值
    pgd_alpha = args.pgd_alpha / 255
    test_batches = data.get_loaders(args.dataset, args.n_final_eval, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)
    
    n_cls = 2 if 'binary' in args.dataset else 10
    
    

    model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0)
    attack_iters, n_restarts = (50, 10)
    for label in ["last", "best", "best_test"]:
        model.load_state_dict(model_saved[label])
        model.eval()
        test_acc_clean, _, _ = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
        test_acc_pgd_rr, _, deltas_pgd_rr = utils.rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
        print('[{}: test on 10k points][iter={}] acc_clean {:.2%}, pgd_rr {:.2%}'.format(label, "???", test_acc_clean, test_acc_pgd_rr))



if __name__ == "__main__":
    main()