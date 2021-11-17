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
### To reproduce the curves, one has to run all the models for seed in {0, 1, 2, 3, 4}, and then average the results
# Train
# python train.py --attack=none --epochs=96 --eval_iter_freq=200 --batch_size=256 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --seed=0
# FGSM
# python train.py --attack=fgsm --eps=8  --attack_init=zero --epochs=96 --eval_iter_freq=200 --batch_size=256 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --seed=0
# PGD-10, eps=8
# python train.py --attack=pgd --pgd_alpha_train=1.6 --pgd_train_n_iters=10 --attack_iters=50 --n_restarts=10 --eps=8  --attack_init=zero --epochs=96 --batch_size=256 --eval_iter_freq=200 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --seed=0
# python train.py --attack=pgd --pgd_alpha_train=6 --pgd_train_n_iters=2 --attack_iters=50 --n_restarts=10 --eps=8  --attack_init=zero --epochs=96 --batch_size=256 --eval_iter_freq=200 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --seed=0
# PGD+GIS-10, eps=8
# python train.py --attack=pgd --pgd_alpha_train=1.6 --pgd_train_n_iters=10 --attack_iters=50 --n_restarts=10 --eps=8  --attack_init=zero --epochs=200 --batch_size=256 --eval_iter_freq=200 --batch_size_eval=1024 --grad_input_sum_coeff=0.8 --eval_early_stopped_model --n_final_eval=-1 --seed=0
# GradAlign
# python train.py --attack=fgsm --eps=2  --attack_init=zero --epochs=1 --batch_size=64 --grad_align_cos_lambda=0.036 --lr_max=0.20 --eval_iter_freq=200 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --seed=0
# FATM AT, 使用PGD-50-10攻击
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --attack_init=zero --epoch=96 --batch_size=128  --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --seed=0
# FATM+GradAlign AT, 使用PGD-50-10攻击 grad_align_cos_lambda太小，晚上设置一个大一点的数重新跑
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --attack_init=zero --epoch=96 --batch_size=128 --grad_align_cos_lambda=0.036  --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --seed=0
# FATM+GradAlign AT, 使用PGD-50-10攻击 grad_align_cos_lambda=0.200, eps=8
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --attack_init=zero --epoch=96 --batch_size=128 --grad_align_cos_lambda=0.200  --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --seed=0
# none+loss_grad_sum,使用PGD-50-10攻击
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --attack_init=zero --epoch=96 --eval_iter_freq=800 --batch_size=128  --lr_max=0.04 --grad_input_sum_coeff=0.2 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --seed=0
# pgd+loss_grad_sum,使用PGD-50-10攻击
# python train.py --attack=pgd --pgd_alpha_train=6 --pgd_train_n_iters=2 --attack_iters=50 --n_restarts=10 --eps=8 --attack_init=zero --epoch=96 --eval_iter_freq=800 --batch_size=128  --lr_max=0.04 --grad_input_sum_coeff=0.2 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --start_eval_epoch=20 --seed=0
# fgsm+rs+loss_grad_sum,使用PGD-50-10攻击
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_init=random --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128  --lr_max=0.04 --grad_input_sum_coeff=0.2 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --start_eval_epoch=30 --seed=0
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_init=random --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128  --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --start_eval_epoch=30 --seed=0

# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128 --grad_input_sum_coeff=0.2 --grad_align_cos_lambda=0.200 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --start_eval_epoch=0 --seed=0
# FSGM+RS+GradAlign-AT
# python train.py --attack=fgsm --fgsm_alpha=2.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.200 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FSGM+RS+G_I_S-AT
# python train.py --attack=fgsm --fgsm_alpha=2.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128 --grad_input_sum_coeff=0.2 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FSGM+RS-AT
# python train.py --attack=fgsm --fgsm_alpha=2.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FSGM-AT
# python train.py --attack=fgsm --attack_init=zero --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FGSM+Discrete_eps-AT
# python train.py --attack=fgsm --attack_init=zero --discrete_eps --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FGSM+RS+Discrete_eps-AT
# python train.py --attack=fgsm --attack_init=random --discrete_eps --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FGSM+GradAlign+Discrete_eps-AT
# python train.py --attack=fgsm --attack_init=zero --discrete_eps --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=10.0 --grad_align_cos_lambda=0.0 --lr_max=0.06 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FGSM+G_I_S+Discrete_eps-AT
# python train.py --attack=fgsm --attack_init=zero --discrete_eps --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.4 --grad_align_cos_lambda=0.0 --lr_max=0.06 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
#FGSM+RS-AT
# python train.py --attack=fgsm --attack_init=random --fgsm_alpha=2.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=400 --batch_size=256 --lr_max=0.04 --batch_size_eval=512 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FGSM+RS+Discrete_eps-AT
# python train.py --attack=fgsm --attack_init=random --discrete_eps --fgsm_alpha=2.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0

# FGSM+Discrete_eps-AT
# python train.py --attack=fgsm --attack_init=zero --discrete_eps --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0

# FGSM+G_I_S-AT
python train.py --attack=fgsm --attack_init=zero --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=9 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FGSM+GradAlign-AT
# python train.py --attack=fgsm --attack_init=zero --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.2 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
# FGSM+MSAT-AT
# python train.py --attack=fgsm --attack_init=zero --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --mast --n_final_eval=-1 --start_eval_epoch=0 --seed=0