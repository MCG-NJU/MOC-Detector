cd src


python3 train.py --K 7 --exp_id Train_K7_rgb_coco_jhmdb_s1 --rgb_model $PATH_TO_SAVE_MODEL --batch_size 63 --master_batch 7 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 25 --lr_step 8,16 --dataset hmdb --split 1 --auto_stop


python3 train.py --K 7 --exp_id Train_K7_flow_coco_jhmdb_s1 --flow_model $PATH_TO_SAVE_MODEL --batch_size 62 --master_batch 6 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --num_epochs 25 --lr_step 12,20 --ninput 5 --dataset hmdb --split 1 --auto_stop




cd ..
