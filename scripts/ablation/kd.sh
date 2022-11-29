
CUDA_VISIBLE_DEVICES=$1 python mixrd.py data/CIFAR10/ \
--dataset-name=CIFAR10 \
--output-dir=output/ablation/kd \
--epochs=800 \
-a=resnet18cifar \
--gpath=checkpoints/cifar10u-cifar-ada-best-fid.pkl \
--syn_ratio=1.0 \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--a_depth=0 \
--num-proj-layers=5 \
--sim_loss_weight=1.0 \
--var_loss_weight=0. \
--cov_loss_weight=0.
