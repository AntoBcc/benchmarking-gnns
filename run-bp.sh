code=experiments_TSP.py 
config=configs/TSP_edge_classification_GatedGCN_100k.json
dataset=tsp30-50_m
#seed=40
seed=41
#seed=42

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 10

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 20

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 50

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 100

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 200

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 500

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 1000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 1500

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 2000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 2500

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 3000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 4000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 5000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 6000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 7000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 8000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 9000

python $code --dataset $dataset --gpu_id 0 --seed $seed --config $config --edge_feat True --train_size 10000
