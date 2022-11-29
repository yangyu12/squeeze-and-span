
CUDA_VISIBLE_DEVICES=$1 python encode.py data/CIFAR100/ \
--output-dir=output/table1/encod_w_syn_cifar100 \
--dataset-name=CIFAR100 \
-a=resnet18cifar \
--epochs=800 \
--gpath=checkpoints/cifar100u-cifar-best-fid4.13.pkl \
--multiprocessing-distributed \
--world-size=1 \
--rank=0

