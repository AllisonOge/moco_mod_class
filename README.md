# moco_mod_class

## Preprocessing

Bucket URLs for experimenting are
- qoherent_external_drive/lnc/recordings
- qoherent_collaborative_share/example_srs_recordings
- qoherent_collaborative_share/recordings

### Usage
```bash
 python data_preparation.py \
 --recordings gs://qoherent_external_drive/lnc/recordings gs://qoherent_collaborative_share/example_srs_recordings gs://qoherent_collaborative_share/recordings \ # bucket urls
 --save_location . \ # current directory
 --name train \ # basename of h5py file
 --suffix lnc-ran-5g_lte_wifi \ # suffix of h5py file
 --length 1024 # length of samples in each example
```

## Research Questions

1. Can Mocov3 training on real world dataset beat supervised model on downstream task?
2. How much masking factor is needed to get the best learned representation?
3. What is the influence of model size on training data using self-supervised learning?
4. What is the influence of more training data on model performance?

**Downstream task (AMC)**


## Training (Pretaining stage)

### Usage
Single node distributed training with 2 GPUs
```bash
OMP_NUM_THREADS=4 torchrun \
 --nproc_per_node=2 \
 -- moco_train.py \
 --epochs 300 \ # number of epochs
 -b 2048 \ # batch size
 --data train_sliced_lnc_riaran.h5 # path to training data
```

## Training (Finetuning stage)

### Usage
```yaml
# config.yaml
batch_size: 256
num_workers: 4
experiment_name: 07dec_frozenmodel
epochs: 50
optimizer: adamw # or sgd
lr: 0.001
weight_decay: 0.00005
momentum: 0.9 # only for sgd
scheduler: reduceonplateau # or cosineannealing
warmup_epochs: 5
dataset:
  train: RML22_train.h5
  val: RML22_val.h5
checkpoint: false
# pretrained: /path/to/pretrained.pth
# freeze: true
classes:
  0: 8psk
  1: am-dsb
  2: bpsk
  3: cpfsk
  4: gfsk
  5: pam4
  6: qam16
  7: qam64
  8: qpsk
  9: wbfm
nclasses: 10

```
```bash
python train.py -c config.yaml
```

## Main Results

> *self-supervised vs supervised learning*

| Metrics | Accuracy (%) | Precision (%) | Recall (%) |
| --- | --- | --- | --- |
| Supervised |  |  |  |
| Ours |  |  |  |

> *Percentage masking factor for augmentation*

| Percent masking factor Pretrained (%) | Fine-tuned (%) |
| --- | --- |
| 10 |  |
| 20 |  |
| 30 |  |
| 40 |  |
| 50 |  |
| 60 |  |
| 70 |  |
| 80 |  |
| 90 |  |

> *Influence of model size*

| Metrics | Accuracy (%) |
| --- | --- |
|   |  Model Size (GB) |
| Supervised |  |
| Ours |  |

> *Influence of more training data*

| Training Data Size (GB) | Percent increase (%) |  Fine-tuned (%) |
| --- | --- | --- |
| 11 | - |  |
| 13 | 18% |  |
| 70 |  536% |  |

## Checkpoints
