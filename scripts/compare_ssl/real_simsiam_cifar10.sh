GPUID=$1

CUDA_VISIBLE_DEVICES=${GPUID} python simsiam.py data/CIFAR10 \
--dataset-name=CIFAR10 \
--epochs=800 \
-a=resnet18cifar \
--lr=0.03 \
--wd=5e-4 \
--dim=2048 \
-b=512 \
-j=32 \
--gpath=checkpoints/cifar10u-cifar-ada-best-fid.pkl \
--syn_ratio=0. \
--num-proj-layers=2 \
--output-dir=output/ssl/real_simsiam_cifar10 \
--fix-pred-lr \
--multiprocessing-distributed \
--world-size=1 \
--rank=0  
