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


## Training

### Usage
```bash

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

> *Influence of more training data*

| Training Data Size (GB) | Percent increase (%) |  Fine-tuned (%) |
| --- | --- | --- |
| 12 | - |  |
| 13 | 8% |  |
|  |   |  |

## Checkpoints
