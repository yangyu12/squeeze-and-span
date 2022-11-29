
CUDA_VISIBLE_DEVICES=$1 python mixrd.py data/CIFAR100/ \
--dataset-name=CIFAR100 \
--output-dir=output/table1/vanilla_distill_w_cifar100 \
--epochs=800 \
-a=resnet18cifar \
--gpath=checkpoints/cifar100u-cifar-best-fid4.13.pkl \
--regex='mapping$' \
--syn_ratio=1.0 \
--augment-syn-data \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--a_depth=0 \
--num-proj-layers=5 \
--sim_loss_weight=1.0 \
--var_loss_weight=0. \
--cov_loss_weight=0.
