GPUID=$1

CUDA_VISIBLE_DEVICES=$GPUID python sns.py data/CIFAR10/ \
--output-dir=output/squeeze_and_span/cifar10 \
--dataset-name=CIFAR10 \
-a=resnet18cifar \
--epochs=800 \
--gpath=checkpoints/cifar10u-cifar-ada-best-fid.pkl \
--syn_ratio=0.5 \
--augment-syn-data \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--a_depth=3 \
--num-proj-layers=5 
