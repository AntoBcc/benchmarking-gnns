
import numpy as np
import torch
import pickle
import time
import os
import argparse
import sys

from torch.utils.data import DataLoader

sys.path.append('../../')
from data.data import LoadData
from data.TSP import TSPDatasetDGL

description = 'create the dataset for the GNN'

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bp',default='',type=str,help='name of folder containing 2-matchings')
parser.add_argument('-hy',default=0,choices=[0,1],type=int,help='1 if using hybrid model, 0 otherwise')
parser.add_argument('-m',default=0,choices=[0,1],type=int,help='1 if using matching as edge features, 0 otherwise')
args = parser.parse_args()

match_file = args.bp
use_matching = bool(args.m)
hybrid = bool(args.hy)

DATASET_NAME = 'tsp30-50'


os.chdir('../../') # go to root folder of the project

start = time.time() 

dataset = TSPDatasetDGL(DATASET_NAME,match_dir=os.path.join('data','TSP','BP',match_file),use_matching=use_matching,hybrid=hybrid) 

print('Time (sec):',time.time() - start) # ~ 30min


print('Node features dimensionality:',dataset.train[0][0].ndata['feat'][0].size(0))
print('Edge features dimensionality:',dataset.train[0][0].edata['feat'][0].size(0))


start = time.time()

desc = ''
if use_matching:
    desc += '_m'
if hybrid:
    desc += '_h'

with open(f'data/TSP/{DATASET_NAME}{desc}.pkl','wb') as f:
    pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
print('Time (sec):',time.time() - start) # ~ 5s
