#!/usr/bin/env bash
### To reproduce the curves, one has to run all the models for seed in {0, 1, 2, 3, 4}, and then average the results

# FGSM
# python train.py --attack=fgsm --eps=1  --attack_init=zero --epochs=5 --eval_iter_freq=200 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --seed=0
# PGD-10, eps=8
# python train.py --attack=pgd --pgd_alpha_train=1.6 --pgd_train_n_iters=10 --attack_iters=50 --n_restarts=10 --eps=8  --attack_init=zero --epochs=96 --batch_size=128 --eval_iter_freq=400 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --seed=0
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
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.200 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --start_eval_epoch=0 --seed=0
# FSGM+RS+G_I_S-AT
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128 --grad_input_sum_coeff=0.2 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --start_eval_epoch=0 --seed=0
# FSGM+RS-AT
# python train.py --attack=fgsm --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=800 --batch_size=128 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=1000 --start_eval_epoch=0 --seed=0
# FGSM+Discrete_eps-AT
python train.py --attack=fgsm --attack_init=zero --discrete_eps --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0
