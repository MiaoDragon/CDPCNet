
#python3 complex_train.py --data_size 80000 --data_minibatch 1000 --batch_size 32 --num_points 2800 \
#--workers 4 --num_epoch 10 --out_path output/ --model complex_simple_model --seed 1 --num_workers 4 \
#--pin_memory 1 --val_batch 10 --save_epoch 2 --model_type simple --data_path /media/arclabdl1/HD1/CDPCNet/complex/

python3 complex_test.py --data_size 80000 --data_minibatch 1000 --batch_size 32 --num_points 2800 \
--workers 4 --num_epoch 10 --out_path output/ --model complex_simple_model --seed 1 --num_workers 4 \
--pin_memory 1 --val_batch 10 --model_type simple --data_path /media/arclabdl1/HD1/YLmiao/CDPCNet/geometry/ \
--start_epoch 10
