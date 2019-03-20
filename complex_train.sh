# generate data
cd dataset
python3 complex_generator.py --data_size 80000 --minibatch 1000 --num_point 2800 --save_path complex/
cd ..

python3 complex_train.py --data_size 80000 --data_minibatch 1000 --batch_size 32 --num_points 2800 \
--workers 4 --num_epoch 10 --out_path output/ --model complex_simple_model --seed 1 --num_workers 4 \
--pin_memory 1 --val_batch 10 --save_epoch 2 --model_type simple

python3 complex_train.py --data_size 80000 --data_minibatch 1000 --batch_size 32 --num_points 2800 \
--workers 4 --num_epoch 10 --out_path output/ --model complex_deep_model --seed 1 --num_workers 4 \
--pin_memory 1 --val_batch 10 --save_epoch 2 --model_type deep
