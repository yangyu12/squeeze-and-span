GPUID=$1

CUDA_VISIBLE_DEVICES=$GPUID python eval_d.py data/CIFAR10 \
--dataset-name=CIFAR10 \
-i=32 \
-a=resnet18cifar \
-b=256 \
--lr=30.0 \
--pretrained=checkpoints/cifar10u-cifar-ada-best-fid.pkl \
--regex="b4.fc"
