cd src


python3 train.py --K 7 --exp_id Train_K7_rgb_coco_resnet18 --rgb_model $PATH_TO_SAVE_MODEL --batch_size 128 --master_batch 16 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 10 --lr_step 5,8 --save_all --arch resnet_18


# -------------------- Evaluation --------------------
# PATH_TO_RGB_MODEL='ucf_resnet18_K7_rgb_coco.pth'

# python3 det.py --task normal --K 7 --gpus 0,1,2,3,4,5,6,7 --batch_size 128 --master_batch 16 --num_workers 16 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --inference_dir $INFERENCE_DIR --flip_test --arch resnet_18
# python3 det.py --task normal --K 7 --gpus 0 --batch_size 1 --master_batch 1 --num_workers 0 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --inference_dir $INFERENCE_DIR --flip_test --arch resnet_18
#
# python3 ACT.py --task frameAP --K 7 --th 0.5 --inference_dir $INFERENCE_DIR
#
# python3 ACT.py --task BuildTubes --K 7 --inference_dir $INFERENCE_DIR
#
# python3 ACT.py --task videoAP --K 7 --th 0.2 --inference_dir $INFERENCE_DIR
# python3 ACT.py --task videoAP --K 7 --th 0.5 --inference_dir $INFERENCE_DIR
# python3 ACT.py --task videoAP --K 7 --th 0.75 --inference_dir $INFERENCE_DIR
# python3 ACT.py --task videoAP_all --K 7 --inference_dir $INFERENCE_DIR

cd ..
