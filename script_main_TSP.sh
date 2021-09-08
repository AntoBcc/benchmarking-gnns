#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_tsp
# tmux detach
# pkill python

# bash script_main_TSP_edge_classification_100k.sh


############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT


############
# TSP - 4 RUNS  
############

seed0=41
seed1=42
seed2=9
seed3=23
code=main_TSP_edge_classification.py 
dataset=tsp30-50

# tmux new -s benchmark_TSP_edge_classification -d
# tmux send-keys "source activate benchmark_gnn" C-m
# tmux send-keys "

#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_MLP_100k.json'

#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GCN_100k.json'

python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_GatedGCN_100k.json' --edge_feat True

# python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_SimpleGatedGCN_100k.json' --edge_feat True
