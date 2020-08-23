# Train

Firstly, download coco pretrained DLA-34 model from [this](https://drive.google.com/file/d/13Sf66b1cEa6ReWMljMmoie4kXA_JBT8T/view?usp=sharing).

We will add more coco pretrained backbone in our [Google drive](https://drive.google.com/drive/folders/1r2uYo-4hL6oOzRARFsYIn5Pu2Lv7VS6m?usp=sharing). COCO pretrained models come from [Centernet](https://drive.google.com/open?id=1px-Xg7jXSC79QqgsD1AAGJQkuf5m0zh_).

Please move pretrained models to `${MOC_ROOT}/experiment/modelzoo`

<br/>

During training,  go to `${PATH_TO_SAVE_MODEL}` and run:

```powershell
tensorboard --logdir=logs_tensorboardX --port=6006
```

 The loss curve will show in `localhost:6006`.

<br/>

The training step will take a long time, we recommend to use `byobu` or `screen` on your server.

<br/>

## Train RGB K=7 on UCF

run

```bash
python3 train.py --K 7 --exp_id Train_K7_rgb_coco --rgb_model $PATH_TO_SAVE_MODEL --batch_size 63 --master_batch 7 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 12 --lr_step 6,8 --save_all

# ==============Args==============
#
# --K              input tubelet length, 7 by default
# --exp_id         your experiment ID
# --rgb_model      path to rgb model
# --batch_size     total batch size 
# --master_batch   batch size in the first gpu
# --lr             initial learning rate, 5e-4 by default (Adam)
# --gpus           gpu list, in our experiments, we use 8 NVIDIA TITAN XP
# --num_workers    total workers
# --num_epoch          max epoch
# --lr_step            epoch for x0.1 learning rate
# --save_all           save each epoch's training model

# [Optional]
# --val_epoch          compute loss on validation set after each epoch
# --pretrain_model		 use `imagenet | coco` pretrained model, `coco` by default
```

If you use `--save_all`, it will save every epoch's training model like `model_[12]_2020-02-12-15-25.pth`. Otherwise it will save last epoch model like `model_last.pth`.

In our coco pretrained experiment, we use model after 12 epoch as RGB model.

In our imagenet pretrained experiment, we use model after 11 epoch as RGB model.

<br/>

## Train FLOW K=7 on UCF

run

```bash
python3 train.py --K 7 --exp_id Train_K7_flow_coco --flow_model $PATH_TO_SAVE_MODEL --batch_size 62 --master_batch 6 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 10 --lr_step 6,8 --ninput 5 --save_all

# additional scripts for flow model
# --ninput 5
```

Add  `--ninput 5`  when training FLOW model .

In our coco pretrained experiment, we use model after 8 epoch as FLOW model.

In our imagenet pretrained experiment, we use model after 10 epoch as FLOW model.

<br/>

<br/>





## Train RGB K=7 on JHMDB

run

```bash
python3 train.py --K 7 --exp_id Train_K7_rgb_coco_jhmdb_s1 --rgb_model $PATH_TO_SAVE_MODEL --batch_size 63 --master_batch 7 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 20 --lr_step 6,8 --dataset hmdb --split 1

# additional scripts for jhmdb
# --dataset hmdb
# --split 1        there are 3 splits
```

In our experiment, we use model after 20 epoch as RGB model for each split.

<br/>

## Train FLOW K=7 on JHMDB

run

```bash
python3 train.py --K 7 --exp_id Train_K7_flow_coco_jhmdb_s1 --flow_model $PATH_TO_SAVE_MODEL --batch_size 62 --master_batch 6 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 20 --lr_step 9,12 --ninput 5 --dataset hmdb --split 1
```

In our experiment, we use model after 20 epoch as FLOW model for each split.

(Recently we notice that training flow model on JHMDB needs more eopch to converge, we will update our results sooner)

<br/>

**But when we reproduce our experiments on another GPU server, we find the training is not stable on JHMDB dataset, so we use `--auto_stop` to eliminate the instability. Because JHMDB dataset is very small and easy to overfit.**

```bash
python3 train.py --K 7 --exp_id Train_K7_rgb_coco_jhmdb_s1 --rgb_model $PATH_TO_SAVE_MODEL --batch_size 63 --master_batch 7 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 25 --lr_step 8,16 --dataset hmdb --split 1 --auto_stop


python3 train.py --K 7 --exp_id Train_K7_flow_coco_jhmdb_s1 --flow_model $PATH_TO_SAVE_MODEL --batch_size 62 --master_batch 6 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 25 --lr_step 12,20 --ninput 5 --dataset hmdb --split 1 --auto_stop
```

It will save the model in `model_best.pth`.

（This may surpass our original results.）

<br/>

<br/>

If you want to reproduce our **ucf-pretrained** results in Supplementary Material, please add `--ucf_pretrain --load_model $PATH_TO_UCF_MODEL`.

`$PATH_TO_UCF_MODEL` is the file path to `dla34_K7_rgb_coco.pth` (for rgb) and `dla34_K7_flow_coco.pth` (for flow).

Also recommend using `--auto stop`.

<br/>

<br/>

## Train RGB K=1 / K=3 / K=5 on UCF

Due to reducing tubelet length K, we can use a large training batch size. So the training scripts change like this:

```bash
python3 train.py --K 5 --exp_id Train_K5_rgb_coco --rgb_model $PATH_TO_SAVE_MODEL --batch_size 108 --master_batch 10 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 13 --lr_step 4,8 --save_all
```

<br/>

<br/>

## Recovery from specific epoch

If the code encounters some running error, we can use following step to recovery training from specific epoch:

```bash
python3 train.py --K 7 --exp_id Train_K7_rgb_coco --rgb_model $PATH_TO_SAVE_MODEL --batch_size 63 --master_batch 7 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 13 --lr_step 6,8 --save_all --load_model ? --start_epoch ?
```

<br/>

For example, if we want to recovery from epoch 4, then we add `--load_model model_[4]_2020-01-20-03-25.pth --start_epoch 4`.

<br/>

## Bash File

We also provide bash file for training. Please refer [train_ucf_k7_dla.sh](../scripts/train_ucf_k7_dla.sh) and [train_jhmdb_k7_dla.sh](../scripts/train_jhmdb_k7_dla.sh).