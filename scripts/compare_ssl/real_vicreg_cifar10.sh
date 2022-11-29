GPUID=$1

CUDA_VISIBLE_DEVICES=$GPUID python vicreg.py data/CIFAR10/ \
--dataset-name=CIFAR10 \
--output-dir=output/ssl/real_vicreg_cifar10 \
-a=resnet18cifar \
--epochs=800 \
--gpath=checkpoints/cifar10u-cifar-ada-best-fid.pkl \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--num-proj-layers=5 \
--syn_ratio=0.
