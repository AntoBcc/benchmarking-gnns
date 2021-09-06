# Informed Machine Learning for the Travelling Salesman Problem: Graph Neural Network (Part 2)

*This work is geared towards a MSc research thesis, for the MSc in Data Science and Business Analytics ([DSBA](http://www.unibocconi.eu/wps/wcm/connect/bocconi/sitopubblico_en/navigation+tree/home/programs/master+of+science/data+science+and+business+analytics)) at Bocconi University. It has been developed with the support and supervision of prof. [CarloLucibello](https://github.com/CarloLucibello).*


#####  README OF THE FORK

This fork shifts the focus on the TSP problem, which is framed as a binary classification problem on each edge of the graph. 
It also adds a smaller TSP dataset (30 to 50 nodes) for faster experimentation. 


The dataset contains 10000 train, 1000 val, 1000 test samples. Each set was generated using the script `data/TSP/generate_TSP.py` using seeds 17,18,19 respectively. 

The objective of the experiments carried out here is to understand whether an **Informed Machine Learning** approach, using external knowledge coming from a Belief Propagation approximation of minimum-weight 2-matching can be helpful in improving *sample efficiency*and *learning efficiency* of a baseline non-informed GNN model.

### Create and activate environment

After cloning this repository:

To run project with GPU (CUDA):
```
# At the root of the project
conda env create -f deeptsp.yml -n deeptsp
conda activate deeptsp
```

This environment is the updated version of the original repo's `environment_gpu.yml.` It uses the most recent stable versions of the required packages available for conda 4.8.3

### Running experiments:

All experiments rely on the file `experiments_TSP.py`, which you can find in the root directory.

To download the data:
```
# At the root of the project
cd data/TSP
bash download-data.sh
```


#### 1. Training size

Experiments are carried out using the GatedGCN model and the configuration file `TSP_edge_classification_GatedGCN_100k.json`.

First, ensure you have created the training, validation and test dataset using `prepare_TSP.py`. At this point, you must decide whether you want to use the the information coming from the minimum-weight perfect 2-matching predictions, obtained by belief propagation, as edge features in the input layer (*BP-enhanced model*) or only at the last stage of the MLP readout (*hybrid model*). 

##### Benchmark model
```
# At the root of the project
cd data/TSP
# 1. Create the dataset:
python  prepare_TSP.py 
#At the root of the project
# 2. Run the model at different training sizes
bash run-benchmark.sh
```


##### BP-enhanced model
```
# At the root of the project
cd data/TSP
# 1. Create the dataset:
python  prepare_TSP.py -bp BP/BPmatch_damp_bayati -m 1
#At the root of the project
# 2. Run the model at different training sizes
bash run-bp.sh 
```

##### Hybrid model
```
# At the root of the project
cd data/TSP
# 1. Create the hybrid dataset:
python  prepare_TSP.py -bp BP/BPmatch_damp_bayati -hy 1
#At the root of the project
# 2. Run the model at different training sizes
bash run-hybrid.sh 
```

Note that it is possible to run the benchmark model on the hybrid dataset, without the need to create a separate one: the base model will simply ignore the BP features. Only the hybrid model will use them in the readout. 

#### 2. Graph size

```
# At the root of the project
cd data/TSP
# create the hybrid dataset, if not created yet
# Split based on graph sizes
python chunk_TSP_data.py  
```

```
#At the root of the project
bash run-chunks.sh 
```

#### 3. Generalization

Training and validating on graph sizes 30-40, testing on graph sizes 41-50. 

```
# At the root of the project
cd data/TSP
# create the hybrid dataset, if not created yet
# Split so that test data has larger-sized graphs
python generalization_data.py  
```

```
#At the root of the project
bash run-generalization.sh 
```


Experiment results will be available as csv files in the `experiment_results` directory. Results for runs of the base model, either with or without BP features, and for the hybrid model will be available in separate csv files organised by initalisation seed (of the sort: `base_model_seedXX.csv` and `hybrid_model_seedXX.csv`), with details about training set size, graph size, running time and performance. More detailed history of any model runs,  will be stored in an `out` directory. 

The plots available in the directory `plots` can be reproduced by running:
```
# At the root of the project
python plot_results.py  
```

All experiments in the end project were carried out with three different seeds: 40, 41, 42. The default seed is set to 41, but it can be changed by simply modifying the corresponding line in each of the bash scripts. 
