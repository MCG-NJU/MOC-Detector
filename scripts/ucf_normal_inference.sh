cd src
# [News] ------  Using '--N 10' and remove '--flip_test will increase the inference speed but get a lower performance.'

python3 det.py --task normal --K 7 --gpus 0,1,2,3,4,5,6,7 --batch_size 94 --master_batch 10 --num_workers 8 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --flow_model ../experiment/result_model/$PATH_TO_FLOW_MODEL --inference_dir $INFERENCE_DIR --flip_test --ninput 5
python3 det.py --task normal --K 7 --gpus 0 --batch_size 1 --master_batch 1 --num_workers 0 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --flow_model ../experiment/result_model/$PATH_TO_FLOW_MODEL --inference_dir $INFERENCE_DIR --flip_test --ninput 5

python3 ACT.py --task frameAP --K 7 --th 0.5 --inference_dir $INFERENCE_DIR

python3 ACT.py --task BuildTubes --K 7 --inference_dir $INFERENCE_DIR

python3 ACT.py --task videoAP --K 7 --th 0.2 --inference_dir $INFERENCE_DIR
python3 ACT.py --task videoAP --K 7 --th 0.5 --inference_dir $INFERENCE_DIR
python3 ACT.py --task videoAP --K 7 --th 0.75 --inference_dir $INFERENCE_DIR
python3 ACT.py --task videoAP_all --K 7 --inference_dir $INFERENCE_DIR
cd ..
