from Functions.Matrix import WPPR, WBFS, WFJ
import Functions.tools as ft
import Functions.Matrix as fm
import networkx as nx
import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
from Functions.tools import LOAD
from tqdm import tqdm 
import pickle

"""
python plot.py

Change the parameters below to correspond to which experiment you want to plot
 - file_name :
    Corresponds to the dataset you want to plot
    only change the last element of the string
    values in : reddit, facebookfriend, epinions
 - W2show : 
    Corresponds to the exposure function you want to plot
    values in : WBFS, WPPR, WFJ
 - strat2show : 
    Corresponds to the Strategies you want to plot
    must be a list
    values inside the list in : 'greedy', ' MPB-all', 'convexity'
 - baseline2show :
    Corresponds to the baselines you want to plot
    must be a list 
    values inside the list in : 'GlobalDemaine', 'SingleSourceDemaine', 'Neurips', 'random_recommendation', 'common_neighbors_recommendation'
 - Q2show
    Corresponds to the target distribution you want to plot
    value in : UniformQ, PreferentialQ

for strat2show and baseline2show, if the experiments are not present nothing will be ploted
"""
file_name = f'./Archive/CIKM results 2024/football'
partition2show = 'ground-truth' #louvain_communities_community_partition, ground-truth
W2show = 'WFJ'
strat2show = ['greedy', ' MPB-all', 'convexity']
baseline2show = ['GlobalDemaine', 'SingleSourceDemaine', 'Neurips', 'random_recommendation', 'common_neighbors_recommendation']
Q2show = 'UniformQ' #PreferentialQ, UniformQ 

colors = {'DescentDiverse':'blue', 'Random':'green', 'Triadic Closure':'red',
          'SpGreedy':'brown', 'Global Diameter Reduction':'orange',
          'Single Source Diameter Reduction': 'purple', 'GreedyDiverse':'pink',
          'MinBoostingDiverse':'cyan'}

def compute_pos_neg_std_columns(data):
    # Convert to a numpy array for easier calculations (assuming each inner list is a row)
    data = np.array(data)
    
    # Calculate the mean of each column
    mean = np.mean(data, axis=0)
    
    # Create empty lists to hold the positive and negative standard deviations for each column
    pos_stds = []
    neg_stds = []
    
    # Loop through each column (axis=1 means column-wise)
    for col in range(data.shape[1]):
        # Get the current column's data
        column_data = data[:, col]
        
        # Calculate positive and negative deviations for the column
        pos_deviations = column_data[column_data > mean[col]] - mean[col]
        neg_deviations = mean[col] - column_data[column_data < mean[col]]
        
        # Calculate the standard deviations
        pos_std = np.sqrt(np.mean(pos_deviations ** 2)) if len(pos_deviations) > 0 else 0
        neg_std = np.sqrt(np.mean(neg_deviations ** 2)) if len(neg_deviations) > 0 else 0
        
        # Append results to the lists
        pos_stds.append(pos_std)
        neg_stds.append(neg_std)
    
    return pos_stds, neg_stds



with open(file_name + '.pkl','rb') as f:
    savedictionary = pickle.load(f)

if W2show == 'WPPR':
    WF2show = WPPR
elif W2show == 'WFJ':
    WF2show = WFJ
elif W2show == 'WBFS':
    WF2show = WBFS



dataname = savedictionary['dataname']
targets = savedictionary['targets']
partition = savedictionary[partition2show]['partition']
graph = LOAD(savedictionary['dataname'])
ft.apply_partition(graph,partition)
if 'continuous' in partition2show:
    P = fm.continuous_P(graph, partition2show.split('-'))
else:
    P = fm.partition_matrix(graph,partition)
k = 50

for experiment in savedictionary[partition2show].keys():



    if W2show in experiment and Q2show in experiment and any([strat in experiment for strat in strat2show]) or\
        any([baseline in experiment for baseline in baseline2show]):
        
        if 'Neurips' in experiment:
            xplabel = 'SpGreedy'
        elif 'Global' in experiment:
            xplabel = 'Global Diameter Reduction'
        elif 'Single' in experiment:
            xplabel = 'Single Source Diameter Reduction'
        elif 'random' in experiment:
            xplabel = 'Random'
        elif 'neighbors' in experiment:
            xplabel = 'Triadic Closure'

        elif 'convexity' in experiment:
            xplabel = 'DescentDiverse'
        elif 'greedy' in experiment:
            xplabel = 'GreedyDiverse'
        elif 'PB' in experiment:
            xplabel = 'MinBoostingDiverse'

        record = []
        for target_node in tqdm(targets):
            COPY = graph.copy()
            Adj = nx.adjacency_matrix(COPY)
            Adj = fm.rowstochastic(Adj).tolil()
            Wm = WF2show(Adj, [target_node,target_node])

            targetarray = np.zeros(shape=(Adj.shape[0],1))
            targetarray[target_node,0] = 1 # np.random.randint(len(graph.nodes))
            targetarray = ss.csr_array(targetarray,dtype=np.int64)
            
            if any([baseline in experiment for baseline in baseline2show]):
                objectiveQ = savedictionary[partition2show][f'{W2show} {strat2show[0]} {Q2show}'][2][target_node]
            else:
                objectiveQ = savedictionary[partition2show][experiment][2][target_node]

            toplot = [fm.OBJECTIVE(P,Wm,targetarray, objectiveQ)[0]]
            for recommended_node in savedictionary[partition2show][experiment][0][target_node]:
                COPY.add_edge(target_node,recommended_node)
                Adj = nx.adjacency_matrix(COPY)
                Adj = fm.rowstochastic(Adj).tolil()
                Wm = WF2show(Adj.tocsc(), [target_node, recommended_node])
                #o, _ = OBJECTIVE(P,Wm,targetarray, objectiveQ)
                o = np.linalg.norm(fm.distribution((P @ Wm @ targetarray).todense()) - objectiveQ)
                toplot.append(o)
            while len(toplot) < k+1:
                toplot.append(toplot[-1])

            record.append(toplot)
        #break
        average = np.mean(record, axis=0)
        plt.plot(average, label = xplabel, color=colors[xplabel])
        #pos, neg = compute_pos_neg_std_columns(record) if record != [] else [np.zeros(50),np.zeros(50)]
        #plt.fill_between(range(51), average - neg, average + pos, color=colors[xplabel], alpha=0.2)
        plt.xlabel('number of added edges')
        plt.ylabel('Objective Function')
        #plt.ylim(1.5,0)
        plt.legend()
        plt.savefig(f'./Results/{dataname} {partition2show} {W2show} {Q2show}.pdf')
        