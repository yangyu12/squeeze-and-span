GPUID=$1

checkpoints=(
    'cifar100u-cifar-000201-fid87.51.pkl' \
    'cifar100u-cifar-000604-fid50.21.pkl' \
    'cifar100u-cifar-001411-fid41.71.pkl' \
    'cifar100u-cifar-002822-fid20.78.pkl' \
    'cifar100u-cifar-005644-fid11.03.pkl' \
    'cifar100u-cifar-011289-fid7.09.pkl' \
    'cifar100u-cifar-022579-fid5.22.pkl' \
    'cifar100u-cifar-045158-fid4.43.pkl' \
)

for CKPT in ${checkpoints[@]}
do

CUDA_VISIBLE_DEVICES=$GPUID python sns.py data/CIFAR100/ \
--dataset-name=CIFAR100 \
--output-dir=output/cifar100_ablation_on_g/ckpt_${CKPT} \
--epochs=800 \
--gpath=checkpoints/${CKPT} \
--augment-syn-data \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--a_depth=3 \
--num-proj-layers=5 \
--sim_loss_weight=10 \
--var_loss_weight=10 

done
