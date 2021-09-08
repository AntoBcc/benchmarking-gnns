import sys
import numpy as np
import pickle
import os

sys.path.append('../../')
from data.data import LoadData

DATASET_NAME = 'tsp30-50_h'
os.chdir('../../')


def split_in_two(data,split_point):
    
    chunks = [[] for i in range(2)]
    
    for g in data:
        nodes = g[0].number_of_nodes()
        if nodes <= split_point:
            chunks[0].append(g)
        else:
            chunks[1].append(g)
                
    return chunks


def generalization_dataset(DATASET_NAME=DATASET_NAME,lb=30,ub=50):

    dataset = LoadData(DATASET_NAME)

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    split_point = int((ub-lb)/2 + lb)

    
    train_chunks = split_in_two(trainset,split_point)
    val_chunks = split_in_two(valset,split_point)
    test_chunks = split_in_two(testset,split_point)

    with open(f'data/TSP/tsp{lb}-{split_point}_gen.pkl','wb') as f:
        pickle.dump([train_chunks[0],test_chunks[1],val_chunks[0]],f)


            
def main():
    generalization_dataset(DATASET_NAME)
    print('Done')
    
    
main()
    
   
