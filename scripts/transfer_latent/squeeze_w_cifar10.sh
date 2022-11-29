
CUDA_VISIBLE_DEVICES=$1 python mixrd.py data/CIFAR10/ \
--dataset-name=CIFAR10 \
--output-dir=output/table1/latent_squeeze_cifar10 \
--epochs=800 \
-a=resnet18cifar \
--gpath=checkpoints/cifar10u-cifar-ada-best-fid.pkl \
--regex='mapping$' \
--syn_ratio=1.0 \
--augment-syn-data \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--a_depth=3 \
--num-proj-layers=5
