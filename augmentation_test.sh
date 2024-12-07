# What percentage of masking gives the best pretrained model performance?

for i in .1 .2 .3 .4 .5 .6 .7 .8 .9; then
    OMP_NUM_THREADS=4 nohup torchrun --nproc_per_node=2 \
    -- \
    moco_train.py \
    --epochs 300 \
    -b 2048 \
    --data ./train_sliced_lnc.h5 \
    -r $i > moco_train_$i.out
done
