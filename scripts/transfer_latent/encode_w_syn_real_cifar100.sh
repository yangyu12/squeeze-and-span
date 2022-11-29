
CUDA_VISIBLE_DEVICES=$1 python encode.py data/CIFAR100/ \
--output-dir=output/table1/encod_w_syn_real_cifar100 \
--dataset-name=CIFAR100 \
-a=resnet18cifar \
--epochs=800 \
--gpath=checkpoints/cifar100u-cifar-best-fid4.13.pkl \
--syn_ratio=0.5 \
--multiprocessing-distributed \
--world-size=1 \
--rank=0
