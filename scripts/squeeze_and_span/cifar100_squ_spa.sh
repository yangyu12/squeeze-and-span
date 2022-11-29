GPUID=$1

CUDA_VISIBLE_DEVICES=$GPUID python sns.py data/CIFAR100/ \
--output-dir=output/squeeze_and_span/cifar100 \
--dataset-name=CIFAR100 \
-a=resnet18cifar \
--epochs=800 \
--gpath=checkpoints/cifar100u-cifar-best-fid4.13.pkl \
--syn_ratio=0.5 \
--augment-syn-data \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--a_depth=3 \
--num-proj-layers=5 \
--sim_loss_weight=10 \
--var_loss_weight=10 
