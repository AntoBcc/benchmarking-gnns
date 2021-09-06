import numpy as np
import pandas as pd

import glob
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns


## PLOTTING FUNCTIONS


def f1_plots(df,variation, dataset,inset=False,vizdir='plots'):
    
    assert variation in ['bp_edge_features','hybrid']
    
    df = df[df.dataset.str.startswith(dataset)]
    
    b = df[df.type=='benchmark'].groupby('size')['test_f1'].agg([np.mean,np.std]).reset_index()
    v = df[df.type==variation].groupby('size')['test_f1'].agg([np.mean,np.std]).reset_index()
    

    v_label = 'Hybrid Gated-GCN' if variation == 'hybrid' else 'BP-Enhanced Gated-GCN'
        
    v_color = 'orangered' if variation == 'hybrid' else 'forestgreen'
    
    fig, ax = plt.subplots(figsize = (11,6))
    ax.plot(v['size'],v['mean'],color=v_color)
    ax.plot(b['size'],b['mean'],color='midnightblue')
    ax.fill_between(v['size'], v['mean'] - v['std'], v['mean'] + v['std'], alpha=0.85,label=v_label,color=v_color)
    ax.fill_between(b['size'], b['mean'] - b['std'], b['mean'] + b['std'], alpha=0.85,label='Gated-GCN',color='midnightblue')
    ax.set_ylim(0.45,1)
    x2 = max(df['size'])
    ax.set_xlim(1,x2)
    ax.set_xlabel('training size')
    ax.set_ylabel('test f1-score')
    title = ax.set_title(f'Gated-GCN vs {v_label} : {dataset} - TEST F1',y=0.92, backgroundcolor='0.9')
    if dataset == 'tsp30-40':
        title = ax.set_title(f'Gated-GCN vs {v_label} : {dataset} (generalization) - TEST F1',y=0.92, backgroundcolor='0.9')

    if inset: 
        axins = inset_axes(ax,4,2, loc=2,bbox_to_anchor=(0.4,0.55),bbox_transform=ax.figure.transFigure)

        axins.plot(v['size'],v['mean'],color=v_color)
        axins.plot(b['size'],b['mean'],color='midnightblue')
        axins.fill_between(v['size'], v['mean'] - v['std'], v['mean'] + v['std'], alpha=0.5,label=v_label,color=v_color)
        axins.fill_between(b['size'], b['mean'] - b['std'], b['mean'] + b['std'], alpha=0.5,label='Gated-GCN',color='midnightblue')
        axins.set_ylim(0.45,0.8)
        axins.set_xlim(1,500)
        axins.set_yticks([i for i in np.arange(start=0.45, stop=0.8, step=0.1)])
        axins.legend(loc=3)


        mark_inset(ax, axins, loc1=2, loc2=4, fc='0.95', ec="0.6")
    else:
        ax.legend(loc=3)

    if vizdir != None:
        if not os.path.exists(vizdir):
            os.makedirs(vizdir)
        plt.savefig(f'{vizdir}/benchmark_{variation}_{dataset}_f1.png',bbox_inches='tight')
        
    #plt.draw()
    #plt.show()
    
    
def epoch_plots(df,variation, dataset,vizdir='plots'):
    
    assert variation in ['bp_edge_features','hybrid']
    
    df = df[df.dataset.str.startswith(dataset)]
    
    b = df[df.type=='benchmark'].groupby('size')['epochs'].agg([np.mean,np.std]).reset_index()
    v = df[df.type==variation].groupby('size')['epochs'].agg([np.mean,np.std]).reset_index()
    

    v_label = 'Hybrid Gated-GCN' if variation == 'hybrid' else 'BP-Enhanced Gated-GCN'
        
    v_color = 'orangered' if variation == 'hybrid' else 'forestgreen'
    
    fig,ax = plt.subplots(figsize=(11,6))
    ax.set_xlabel('training size')
    ax.set_ylabel('epochs')
    title = ax.set_title(f'Gated-GCN vs {v_label} : {dataset} - EPOCHS',y=0.92, backgroundcolor='0.9')
    if dataset == 'tsp30-40':
        title = ax.set_title(f'Gated-GCN vs {v_label} : {dataset} (generalization) - EPOCHS',y=0.92, backgroundcolor='0.9')

    ax.errorbar(b['size'], b['mean'], yerr=b['std'],label='Gated-GCN',color='midnightblue')

    ax.errorbar(v['size'], v['mean'], yerr=v['std'],label=v_label,color=v_color)

    ax.set_ylim(100,450)
    x2 = max(df['size'])
    ax.set_xlim(1,x2)

    plt.legend(loc=0, bbox_to_anchor=(0.98,0.85));
    
    if vizdir != None:
        if not os.path.exists(vizdir):
            os.makedirs(vizdir)
        plt.savefig(f'{vizdir}/benchmark_{variation}_{dataset}_epochs.png',bbox_inches='tight')


    #plt.draw()
    #plt.show()


def main():

    #reading in the data 
    res_dir = 'experiment_results'
    files = glob.glob(res_dir + "/*.csv")
    header = ['dataset','type','size','test_f1','train_f1','epochs','time_per_epoch','tot_time']

    dfs = []

    for file in files:
        temp = pd.read_csv(file, names=header)
        dfs.append(temp)

    df = pd.concat(dfs)

    
    f1_plots(df,'bp_edge_features','tsp30-50',inset=True)
    f1_plots(df,'hybrid','tsp30-50',inset=True)

    f1_plots(df,'hybrid','tsp30-35',inset=True)
    f1_plots(df,'hybrid','tsp36-40',inset=True)
    f1_plots(df,'hybrid','tsp41-45',inset=True)
    f1_plots(df,'hybrid','tsp46-50',inset=True)

    f1_plots(df,'hybrid','tsp30-40',inset=True)

    epoch_plots(df,'bp_edge_features','tsp30-50')
    epoch_plots(df,'hybrid','tsp30-50')

    epoch_plots(df,'hybrid','tsp30-35')
    epoch_plots(df,'hybrid','tsp36-40')
    epoch_plots(df,'hybrid','tsp41-45')
    epoch_plots(df,'hybrid','tsp46-50')

    epoch_plots(df,'hybrid','tsp30-40')


main()




