GPUID=$1
PORT=$2

CUDA_VISIBLE_DEVICES=$GPUID python vicreg.py data/CIFAR100/ \ --dataset-name=CIFAR100 \
--output-dir=output/ssl/syn_vicreg_cifar100 \
-a=resnet18cifar \
--epochs=800 \
--gpath=checkpoints/cifar100u-cifar-best-fid4.13.pkl \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--num-proj-layers=5 \
--syn_ratio=1.
