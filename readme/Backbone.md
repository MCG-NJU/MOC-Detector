# Backbone

To be continued...

| Backbone  |  K   | Modality | Flip | FrameAP @0.5 | VideoAP @0.2 \| @0.5 \| @0.75 \| 0.5:0.95 | FPS  |                           Download                           |
| :-------: | :--: | :------: | :--: | :----------: | :---------------------------------------: | :--: | :----------------------------------------------------------: |
|  DLA-34   |  7   |   RGB    |  ✓   |    73.14     |     78.81 \| 51.02 \| 27.05 \| 26.51      |  29  | [model](https://drive.google.com/file/d/1wQQC4btMxdOp5vAx9DxP3u2z-RejeLPm/view?usp=sharing) |
|           |      |          |  ✗   |    72.05     |     78.23 \| 50.77 \| 26.10 \| 26.16      |  53  |                              ⬆️                               |
|           |      |          |      |              |                                           |      |                                                              |
| ResNet-18 |  7   |   RGB    |  ✓   |    70.36     |     76.31 \| 50.03 \| 25.66 \| 25.72      |  50  | [model](https://drive.google.com/file/d/1_Y5R64euVNtpyAzGFP3mjxSM2p9Q0Z_J/view?usp=sharing) |
|           |      |          |  ✗   |    68.63     |     76.70 \| 49.31 \| 24.63 \| 25.11      |  85  |                              ⬆️                               |

*(All experiments validate on UCF101-24 test split with coco pretrain)*

Online FPS tests on a single NVIDIA TITAN XP with `--batch_size  1`. 

<br/>

## Train RGB K=7 on UCF for ResNet-18

Firstly, download coco pretrained ResNet-18 model from [this](https://drive.google.com/file/d/1PgI52M_N9NYipMm9kl8eMCUQGGDHvmVK/view?usp=sharing).

We get this pretrained model from [Centernet](https://drive.google.com/open?id=1px-Xg7jXSC79QqgsD1AAGJQkuf5m0zh_), which adds three up-convolutional layers to obtain a higher-resolution output.

Please move this pretrained model to `${MOC_ROOT}/experiment/modelzoo`

<br/>

Then, run

```bash
python3 train.py --K 7 --exp_id Train_K7_rgb_coco_resnet18 --rgb_model $PATH_TO_SAVE_MODEL --batch_size 128 --master_batch 16 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 10 --lr_step 5,8 --save_all --arch resnet_18
```

Don't forget to add `--arch resnet_18`.

<br/>

## Evaluate RGB K=7 on UCF for ResNet-18

Download our result model from [this](https://drive.google.com/file/d/1_Y5R64euVNtpyAzGFP3mjxSM2p9Q0Z_J/view?usp=sharing). Then run:

```bash
python3 det.py --task normal --K 7 --gpus 0,1,2,3,4,5,6,7 --batch_size 128 --master_batch 16 --num_workers 16 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --inference_dir $INFERENCE_DIR --flip_test --arch resnet_18
python3 det.py --task normal --K 7 --gpus 0 --batch_size 1 --master_batch 1 --num_workers 0 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --inference_dir $INFERENCE_DIR --flip_test --arch resnet_18

python3 ACT.py --task frameAP --K 7 --th 0.5 --inference_dir $INFERENCE_DIR

python3 ACT.py --task BuildTubes --K 7 --inference_dir $INFERENCE_DIR

python3 ACT.py --task videoAP --K 7 --th 0.2 --inference_dir $INFERENCE_DIR
python3 ACT.py --task videoAP --K 7 --th 0.5 --inference_dir $INFERENCE_DIR
python3 ACT.py --task videoAP --K 7 --th 0.75 --inference_dir $INFERENCE_DIR
python3 ACT.py --task videoAP_all --K 7 --inference_dir $INFERENCE_DIR
```

Don't forget to add `--arch resnet_18`.

<br/>

## Bash File

We also provide bash file for training. Please refer [train_ucf_k7_resnet18.sh](../scripts/train_ucf_k7_resnet18.sh).