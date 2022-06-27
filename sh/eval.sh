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
TIMESTAMP=235551
MODEL_LABELS=last
# python eval.py --dataset=tiny_imagenet --timestamp=$TIMESTAMP --model=resnet34 --naive_eval --model_labels=$MODEL_LABELS 
eps=8
# 这里是eval.py fgsm_alpha的单位直接是像素值，和train.py不一样
# for fgsm_alpha in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0
for fgsm_alpha in 8.0
# for fgsm_alpha in 16.0
# for ((fgsm_alpha=1;fgsm_alpha<1;fgsm_alpha++))
do
    echo "fgsm_alpha is $fgsm_alpha"
    echo "1 eps is $eps"
    if [ $(echo "$fgsm_alpha > $eps"|bc) -eq 1 ]
    then
        echo "fgsm_alpha is so great"
        eps=$fgsm_alpha
    fi
    echo "2 eps is $eps"
    python eval.py --dataset=tiny_imagenet --timestamp=$TIMESTAMP --model=resnet34 --model_labels=$MODEL_LABELS --eps=$eps --fgsm_eval --fgsm_alpha=$fgsm_alpha
    
done
# pgd测试
# eps=8
# python eval.py --dataset=tiny_imagenet --timestamp=$TIMESTAMP --model=resnet34 --model_labels=$MODEL_LABELS --eps=$eps --pgd_eval --pgd_alpha=2