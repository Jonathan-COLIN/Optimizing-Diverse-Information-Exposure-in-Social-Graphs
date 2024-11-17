import networkx as nx
import pickle
import time
import datetime
import os
import argparse
from tqdm import tqdm

import Functions.tools as ft
import Functions.Partitions as fp
import Functions.Matrix as fm
import Functions.Recommenders as fr
LOAD = ft.LOAD

parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('--data', type=str, help='data name')
parser.add_argument('--p', type=str, help='partition')
parser.add_argument('--w', type=str, help='Wfunction')
parser.add_argument('--s', type=str, help='strategy')
parser.add_argument('--q', type=str, help='target distribution')
parser.add_argument('--m', type=str, help='model')
parser.add_argument('--c', type=str, help='continuous partition attribute name', default='')

"""
python main.py --data polbooks --p louvain --w WPPR --m recommendMain --s greedy --q pref --c genre

--data  : karate polbooks football emails college ml-100k facebook brexit astro vaxnovax reddit facebookfriend epinions googleplus amazon youtube
--p     : louvain, gr(ground-truth), c
--w     : WBFS, WPPR, WFJ
--m     : recommendMain, GlobalDemaine, SingleSourceDemaine, Neurips, common_neighbors_recommendation, EffectEstimation, random_recommendation
--s     : greedy, convexity
            MPB, RMPB, BMPB
                all, ssbr, ssdb, sstb
--q     : uni, pref
--c     : genre(ml-100k), polarity(brexit,vaxnovax)
"""


if __name__ == "__main__":

    args = parser.parse_args()

    dataname = args.data
    Wfunction_name = args.w
    model = getattr(fr, args.m)
    strategy = args.s
    k = 50

    graph = LOAD(dataname)
    file_name = dataname
    savedictionary = {'dataname':dataname, 'k':k}

    if args.p == 'louvain':
        partitioname = fp.louvain_communities_community_partition.__name__
    elif args.p == 'gr':
        partitioname = 'ground-truth'
    elif args.p == 'c' and args.c is not None:
        partitioname = 'continuous-'+args.c

    if args.q == 'uni':
        Qfunction = fr.UniformQ
    elif args.q == 'pref':
        Qfunction = fr.PreferentialQ   

    if Wfunction_name == 'WPPR':
        Wfunction = fm.WPPR_bicg
    if Wfunction_name == 'WFJ':
        Wfunction = fm.WFJ_bicg
    if Wfunction_name == 'WBFS':
        Wfunction = fm.WBFS_bicg


    ################################################
    ######### Checking if logs exist
    ######### Else
    ######### Creating logs
    ################################################
    if file_name+'.pkl' in os.listdir('./Saves/'):
        print('found existing logs')
        with open(f'./Saves/{file_name}.pkl','rb') as f:
            baseDictionary = pickle.load(f)
            if 'targets' in baseDictionary.keys():
                targets = baseDictionary['targets']
            else:
                targets = fr.selectTargets(graph, 30)
            savedictionary['targets'] = targets


            if partitioname in baseDictionary.keys():
                partition = baseDictionary[partitioname]['partition']
                if args.p == 'c' and args.c is not None:
                    P = fm.continuous_P(graph,args.c)
                else:
                    P = fm.partition_matrix(graph,partition)
                savedictionary[partitioname] = baseDictionary[partitioname]
            else:
                partition, P = fm.produce_partition(graph,partitioname,args.c)
                savedictionary[partitioname] = {'partition':partition}

            for element in baseDictionary.keys():
                if element not in ['dataname', 'k', 'targets']:
                    savedictionary[element] = baseDictionary[element]
    else:
        print('no logs found')
        targets = fr.selectTargets(graph, 30)
        partition, P = fm.produce_partition(graph,partitioname,args.c)
        savedictionary['targets'] = targets
        savedictionary[partitioname] = {'partition': partition}
        with open(f'./Saves/{file_name}.pkl','wb+') as f:
            pickle.dump(savedictionary,f)

    print(dataname, Wfunction_name, model.__name__, strategy, k, Qfunction.__name__, partitioname)
    print(f'Starting at {datetime.datetime.now()}')
    #==================================================================================================
    #==================================================================================================
    #==================================================================================================


    if model.__name__ == 'recommendMain':
        experiment_name = f'{Wfunction_name} {strategy} {Qfunction.__name__}'

        if experiment_name in savedictionary[partitioname]:
            print('experiment found')
            recommendations = savedictionary[partitioname][experiment_name][0]
            times = savedictionary[partitioname][experiment_name][1]
            objective_distributions = savedictionary[partitioname][experiment_name][2]
            #checking if all distribution have been computed
            if len(objective_distributions.keys()) < len(targets):
                print('missing target distribution')
                for u in targets:
                    if u not in objective_distributions.keys():
                        objective_distributions[u] = Qfunction(P,Wfunction(nx.adjacency_matrix(graph), P, u, fm.UniformQ(P,[],u)), u)

            measured_objective = savedictionary[partitioname][experiment_name][3]
        else:
            print('experiment not found')
            recommendations = {u:[] for u in targets}
            times = {u:[] for u in targets}
            print('computing target distributions')
            objective_distributions = {u:Qfunction(P,Wfunction(nx.adjacency_matrix(graph), P, u, fm.UniformQ(P,[],u)), u) for u in tqdm(targets)}
            measured_objective = {u:[] for u in targets}


            with open(f'./Saves/{file_name}.pkl','rb') as f:
                baseDictionary = pickle.load(f)
                
                if partitioname in baseDictionary.keys():
                    savedictionary = baseDictionary
                else:
                    savedictionary[partitioname] = {'partition': partition}
                savedictionary[partitioname][experiment_name] = [recommendations, times, objective_distributions, measured_objective]
            with open(f'./Saves/{file_name}.pkl','wb') as f:
                pickle.dump(savedictionary,f)

        replace = all([ len(rec) >= k for rec in recommendations.values()])
        print('starting optimization')
        for u in tqdm(targets):
            if replace:
                pass
            elif len(recommendations[u]) >= k:
                continue
            targetQ = objective_distributions[u]

            start = time.time()
            nodes, scores = model(graph, u, k, Wfunction, P, strategy=strategy, targetQ=targetQ)
            end = time.time()

            recommendations[u] = nodes
            times[u] = end - start
            measured_objective[u] = scores

            savedictionary[partitioname][experiment_name] = [recommendations, times, objective_distributions, measured_objective]

            with open(f'./Saves/{file_name}.pkl','rb') as f:
                baseDictionary = pickle.load(f)
               
                savedictionary = baseDictionary
                savedictionary[partitioname][experiment_name] = [recommendations, times, objective_distributions, measured_objective]
            with open(f'./Saves/{file_name}.pkl','wb') as f:
                pickle.dump(savedictionary,f)
    else:
        experiment_name = f'{model.__name__} {Wfunction_name} {Qfunction.__name__}'


        if experiment_name in savedictionary[partitioname]:
            print('experiment found')
            recommendations = savedictionary[partitioname][experiment_name][0]
            times = savedictionary[partitioname][experiment_name][1]
            objective_distributions = savedictionary[partitioname][experiment_name][2]
            measured_objective = savedictionary[partitioname][experiment_name][3]
        else:
            print('experiment not found')
            recommendations = {u:[] for u in targets}
            times = {u:[] for u in targets}
            objective_distributions = {u:Qfunction(P,Wfunction(nx.adjacency_matrix(graph), P, u, fm.UniformQ(P,[],u)), u) for u in targets}
            measured_objective = {u:{} for u in targets}

            with open(f'./Saves/{file_name}.pkl','rb') as f:
                baseDictionary = pickle.load(f)
                
                savedictionary = baseDictionary
                savedictionary[partitioname][experiment_name] = [recommendations, times, objective_distributions, measured_objective]
            with open(f'./Saves/{file_name}.pkl','wb') as f:
                pickle.dump(savedictionary,f)

        replace = all([ len(rec) >= k for rec in recommendations.values()])
        for u in tqdm(targets):
            if replace:
                pass
            elif len(recommendations[u]) >= k:
                continue
            targetQ = objective_distributions[u]

            start = time.time()
            l = model(graph, u, k)
            end = time.time()

            recommendations[u] = l
            times[u] = end - start
            measured_objective[u] = fm.OBJECTIVE_list(graph, P, Wfunction, u, targetQ, l)

            savedictionary[partitioname][experiment_name] = [recommendations, times, objective_distributions, measured_objective]

            with open(f'./Saves/{file_name}.pkl','rb') as f:
                baseDictionary = pickle.load(f)
                
                savedictionary = baseDictionary
                savedictionary[partitioname][experiment_name] = [recommendations, times, objective_distributions, measured_objective]
            with open(f'./Saves/{file_name}.pkl','wb') as f:
                pickle.dump(savedictionary,f)

    #==================================================================================================
    #==================================================================================================
    #==================================================================================================
    f.close()
    print('ended at : ',datetime.datetime.now())