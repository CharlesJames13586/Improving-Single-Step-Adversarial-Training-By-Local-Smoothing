# Improving Single-Step Adversarial Training By Local Smoothing

## Abstract
The excellent model obtained through natural data training in deep learning is easily tampered with by adversarial examples. After discovering that, adversarial training has become the best way to defend against adversarial attacks and improve the robustness of the model. Since it is expensive to frequently calculate adversarial examples in each epoch during the training process, most people prefer to choose a single-step adversarial training method. However, the single-step adversarial training method will cause catastrophic overfitting and make the model lose robustness forever. In this paper, we explain adversarial training from the perspective of data augmentation, using artificial binary data to explore the reason for the occurrence of this overfitting. We propose two methods (VFSAT and GradSum) to prevent the overfitting in term of local smoothing and improve the robustness of the model obtained by single-step adversarial training. Simultaneously, experiments on CIFAR-10 and Tiny ImageNet datasets were constructed and the proof that single-step adversarial training could also resist multi-step adversarial attacks was derived.

## About the paper


## Code

### Training code
Training ResNet-18 using FGSM+VFSAT on CIFAR-10 can be done as follows: `python train.py --attack=fgsm --attack_init=zero --discrete_eps --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=16 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=0.0 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0`

Training ResNet-18 using FGSM+GradSum on CIFAR-10 can be done as follows: `python train.py --attack=fgsm --attack_init=zero --fgsm_alpha=1.0 --attack_iters=50 --n_restarts=10 --eps=8 --epoch=96 --eval_iter_freq=200 --batch_size=256 --grad_input_sum_coeff=10 --grad_align_cos_lambda=0.0 --lr_max=0.04 --batch_size_eval=1024 --eval_early_stopped_model --n_final_eval=-1 --start_eval_epoch=0 --seed=0`

As Tiny ImageNet is much larger, we suggest using ResNet-34 for better fitting performance. This requires more computational resources, so we used multied GPUs for training the launch command can be found under the folder [/sh](/sh).