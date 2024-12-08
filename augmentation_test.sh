# What percentage of masking gives the best pretrained model performance?

for i in $(seq 0.1 0.1 0.9); do
    echo "Starting training with masking ratio $i"
    OMP_NUM_THREADS=4 nohup torchrun --nproc_per_node=2 \
    moco_train.py \
    --epochs 100 \
    -b 2048 \
    --data ./train_sliced_lnc.h5 \
    -r $i > moco_train_${i}.out
done
