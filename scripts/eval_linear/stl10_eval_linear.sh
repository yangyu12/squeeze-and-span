GPUID=$1
PRETRAIN=$2

CUDA_VISIBLE_DEVICES=$GPUID python eval_classification.py data/STL10 \
--dataset-name=STL10 \
-i=64 \
-a=resnet18 \
--pretrained=$PRETRAIN \
--lars
