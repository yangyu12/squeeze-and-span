GPUID=$1

CUDA_VISIBLE_DEVICES=$GPUID python vicreg.py data/STL10/ \
--dataset-name=STL10 \
--output-dir=output/ssl/real_vicreg_stl10 \
-i=64 \
--epochs=200 \
--gpath=checkpoints/stl10u-my128-best-fid20.86.pkl \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--num-proj-layers=5 \
--lr=0.05 \
--wd=1e-4 \
--syn_ratio=0.
