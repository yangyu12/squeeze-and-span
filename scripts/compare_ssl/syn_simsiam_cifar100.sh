GPUID=$1

CUDA_VISIBLE_DEVICES=${GPUID} python simsiam.py data/CIFAR100 \
--dataset-name=CIFAR100 \
--epochs=800 \
-a=resnet18cifar \
--lr=0.03 \
--wd=5e-4 \
--dim=2048 \
-b=512 \
-j=32 \
--gpath=checkpoints/cifar100u-cifar-best-fid4.13.pkl \
--syn_ratio=1. \
--num-proj-layers=2 \
--output-dir=output/ssl/syn_simsiam_cifar100 \
--fix-pred-lr \
--multiprocessing-distributed \
--world-size=1 \
--rank=0  
