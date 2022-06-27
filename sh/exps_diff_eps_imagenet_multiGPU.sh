#!/usr/bin/env bash
# 指定gpu_id

export CUDA_VISIBLE_DEVICES="1,2,3"
# FGSM
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_multiGPU.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=10 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=150 --eval_epoch_freq=10 --batch_size=64 --batch_size_eval=128 --model=resnet34 --lr_max=0.05 --n_final_eval=-1 --seed=1998
# FGSM 续写
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=7 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --eval_iter_freq=8000 --batch_size=128 --batch_size_eval=128 --load_model=095142 --model=resnet18 --n_final_eval=-1 --seed=0
# PGD-2
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_multiGPU.py --attack=pgd --pgd_alpha_train=4 --pgd_train_n_iters=2 --attack_iters=7 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=150 --eval_epoch_freq=10 --batch_size=32 --batch_size_eval=128 --model=resnet34 --lr_max=0.05 --n_final_eval=-1 --seed=0
# GradSum
# python -m torch.distributed.launch --nproc_per_node=5 --nnodes=1 train_multiGPU.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=7 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=96 --eval_iter_freq=6250 --batch_size=32 --model=resnet34 --lr_max=0.05 --grad_input_sum_coeff=10 --n_final_eval=-1  --n_eval_every_k_iter=320 --seed=0
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_multiGPU.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=7 --dataset=tiny_imagenet --eps=8  --attack_init=zero --epochs=150 --eval_epoch_freq=10 --batch_size=32 --batch_size_eval=128 --model=resnet34 --lr_max=0.05 --grad_input_sum_coeff=10 --n_final_eval=-1 --seed=2022
# VFSAT
python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 train_multiGPU.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=7 --dataset=tiny_imagenet --discrete_eps --eps=8  --attack_init=zero --epochs=150 --eval_epoch_freq=10 --batch_size=32 --batch_size_eval=128 --model=resnet34 --lr_max=0.05 --grad_input_sum_coeff=0 --n_final_eval=-1 --seed=2022