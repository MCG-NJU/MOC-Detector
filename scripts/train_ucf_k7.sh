cd src



python3 train.py --K 7 --exp_id Train_K7_rgb_coco --rgb_model $PATH_TO_SAVE_MODEL --batch_size 63 --master_batch 7 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 12 --lr_step 6,8 --save_all

python3 train.py --K 7 --exp_id Train_K7_flow_coco --flow_model $PATH_TO_SAVE_MODEL --batch_size 62 --master_batch 6 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 10 --lr_step 6,8 --ninput 5 --save_all




cd ..
