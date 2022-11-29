GPUID=$1
PRETRAIN=$2

CUDA_VISIBLE_DEVICES=$GPUID python eval_classification.py data/CIFAR10 \
--dataset-name=CIFAR10 \
-i=32 \
-a=resnet18cifar \
-b=256 \
--lr=30.0 \
--pretrained=$PRETRAIN
