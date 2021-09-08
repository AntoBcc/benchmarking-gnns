import sys
import numpy as np
import pickle
import os

sys.path.append('../../')
from data.data import LoadData

DATASET_NAME = 'tsp30-50_h'
os.chdir('../../')

def split_by_size(data,lb,ub,splits):
    
    n_splits = len(splits)
    
    chunks = [[] for i in range(n_splits)]
    
    for g in data:
        nodes = g[0].number_of_nodes()
        for s in range(n_splits):
            if nodes in splits[s]:
                chunks[s].append(g)
                
    return chunks


def data_chunks(DATASET_NAME=DATASET_NAME,lb=30,ub=50,n_splits=4):
    
    dataset = LoadData(DATASET_NAME)
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    
    r = range(lb,ub+1)
    splits = np.array_split(r,n_splits)
    
    ids = [f'{splits[i][0]}-{splits[i][-1]}' for i in range(n_splits)]
    
    train_chunks = split_by_size(trainset,lb,ub,splits=splits)
    val_chunks = split_by_size(valset,lb,ub,splits=splits)
    test_chunks = split_by_size(testset,lb,ub,splits=splits)
    
    for i in range(n_splits):
        with open(f'data/TSP/tsp_{ids[i]}.pkl','wb') as f:
            pickle.dump([train_chunks[i],val_chunks[i],test_chunks[i]],f)
            
            
def main():
    data_chunks(DATASET_NAME)
    print('Done')
    
    
main()
    
   
