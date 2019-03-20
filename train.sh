# generate data
cd dataset
python3 geometry_generator.py --data_size 20000 --minibatch 1000 --num_point 2800 --save_path geometry/
cd ..
python3 train.py --data_size 20000 --data_minibatch 1000 --batch_size 32 --num_points 2800 \
--workers 4 --num_epoch 250 --out_path output/ --model geo_model --seed 1 --num_workers 4 \
--pin_memory 1 --val_batch 10 --save_epoch 5
