GPUID=$1

CUDA_VISIBLE_DEVICES=$GPUID python sns.py data/STL10/ \
--output-dir=output/squeeze_and_span/stl10 \
--dataset-name=STL10 \
-i=64 \
--syn_ratio=0.5 \
--epochs=200 \
--gpath=checkpoints/stl10u-my128-best-fid20.86.pkl \
--augment-syn-data \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--a_depth=3 \
--num-proj-layers=5 \
--lr=0.05 \
--wd=1e-4 
