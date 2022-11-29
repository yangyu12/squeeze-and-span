
CUDA_VISIBLE_DEVICES=$1 python encode.py data/CIFAR10/ \
--output-dir=output/table1/encod_w_syn_real_cifar10 \
--dataset-name=CIFAR10 \
-a=resnet18cifar \
--epochs=800 \
--gpath=checkpoints/cifar10u-cifar-ada-best-fid.pkl \
--syn_ratio=0.5 \
--multiprocessing-distributed \
--world-size=1 \
--rank=0
