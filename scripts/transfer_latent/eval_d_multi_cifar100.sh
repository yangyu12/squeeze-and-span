GPUID=$1

CUDA_VISIBLE_DEVICES=$GPUID python eval_d.py data/CIFAR100 \
--dataset-name=CIFAR100 \
-i=32 \
-a=resnet18cifar \
-b=256 \
--lr=30.0 \
--pretrained=checkpoints/cifar100u-cifar-best-fid4.13.pkl
