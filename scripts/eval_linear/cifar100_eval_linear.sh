GPUID=$1
PRETRAIN=$2

CUDA_VISIBLE_DEVICES=$GPUID python eval_classification.py data/CIFAR100 \
--dataset-name=CIFAR100 \
-i=32 \
-a=resnet18cifar \
-b=256 \
--lr=30.0 \
--pretrained=$PRETRAIN
