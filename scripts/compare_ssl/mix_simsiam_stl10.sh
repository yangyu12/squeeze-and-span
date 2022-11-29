GPUID=$1

CUDA_VISIBLE_DEVICES=${GPUID} python simsiam.py data/STL10 \
--dataset-name=STL10 \
--output-dir=output/ssl/mix_simsiam_stl10 \
-i=64 \
--epochs=200 \
-a=resnet18 \
--lr=0.05 \
--wd=1e-4 \
--dim=2048 \
-b=512 \
-j=32 \
--num-proj-layers=3 \
--fix-pred-lr \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--syn_ratio=0.5 \
--gpath=checkpoints/stl10u-my128-best-fid20.86.pkl 
