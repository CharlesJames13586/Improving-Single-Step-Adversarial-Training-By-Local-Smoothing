#!/usr/bin/env bash
# 指定gpu_id
echo "传递的参数个数为 $#"
if [ $# -gt 0 ];
then
    export CUDA_VISIBLE_DEVICES=$1
    echo "指定在gpu $1 上运行代码"
else
    echo "在默认gpu上运行代码"
fi
# FGSM
#  python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=7 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --eval_iter_freq=20000 --batch_size=64 --batch_size_eval=128 --model=resnet34 --lr_max=0.05 --n_final_eval=-1 --seed=0
# FGSM 续写
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=7 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --eval_iter_freq=8000 --batch_size=128 --batch_size_eval=128 --load_model=095142 --model=resnet18 --n_final_eval=-1 --seed=0
# PGD-2
# python train.py --attack=pgd --pgd_alpha_train=4 --pgd_train_n_iters=2 --attack_iters=7  --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --batch_size=128 --eval_iter_freq=8000 --batch_size_eval=128 --model=resnet18 --n_final_eval=-1 --seed=0
# python train.py --attack=pgd --pgd_alpha_train=4 --pgd_train_n_iters=2 --attack_iters=7  --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --batch_size=128 --eval_iter_freq=40000 --batch_size_eval=128 --load_model=153408 --model=resnet18 --n_final_eval=-1 --seed=0
# python train.py --attack=pgd --pgd_alpha_train=4 --pgd_train_n_iters=2 --attack_iters=7  --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --batch_size=64 --eval_iter_freq=40000 --batch_size_eval=128 --model=resnet34 --lr_max=0.05 --n_final_eval=-1 --seed=0
# GradSum
python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=7 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --eval_iter_freq=40000 --batch_size=64 --batch_size_eval=128 --model=resnet34 --lr_max=0.05 --grad_input_sum_coeff=10 --n_final_eval=-1 --seed=0