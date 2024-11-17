import numpy as np
import networkx as nx
from sklearn.cluster import KMeans

default_numbers_of_parts = 5
number_of_parts = 0

'''
Randomly assigns partitions to nodes in the graph
'''
def random_partition(graph, nb=default_numbers_of_parts):
    partition = {i:[] for i in range(nb)}
    global number_of_parts
    number_of_parts = default_numbers_of_parts
    for node in graph.nodes:
        part = np.random.randint(0,nb)
        partition[part].append(node)
    return partition

'''
Assigns partitions to nodes according to greedy modularity partition
'''
def greedy_modularity_community_partition(graph):
    communities = nx.algorithms.community.greedy_modularity_communities(graph, cutoff = 8, best_n = 10)
    global number_of_parts
    number_of_parts = len(communities)
    return {i:list(communities[i]) for i in range(len(communities))}


'''
Assigns partitions to nodes according to Louvain partition
'''
def louvain_communities_community_partition(graph):
    communities = nx.community.louvain_communities(graph)

    global number_of_parts
    number_of_parts = len(communities)

    partition = {}
    i = 0
    for community in communities:
        for node in community:
            if i in partition:
                partition[i].append(node)
            else:
                partition[i] = [node]
        i += 1

    return partition

'''
Assigns partitions to nodes according to their degree
'''
def degree_partition(graph):
    partition = {}
    for node,degree in graph.degree:
        if degree in partition:
            partition[degree].append(node)
        else:
            partition[degree] = [node]
    global number_of_parts
    number_of_parts = len(partition)
    i =0
    result = {}
    for k,v in partition.items():
        result[i] = v
        i += 1
    return result


'''
Assigns partitions to nodes by clustering them according to their pagerank values
'''
def pagerank_clustering_partition(graph, nb=default_numbers_of_parts):
    partition = {i:[] for i in range(nb)}
    prs = nx.pagerank(graph)
    clusterer = KMeans(n_clusters=nb, n_init='auto')
    indices = clusterer.fit_predict(np.array([v for k,v in prs.items()]).reshape(-1,1))
    dic = {k:p for k,p in zip(graph.nodes,indices)}
    for k,v in dic.items():
        partition[v].append(k)
    global number_of_parts
    number_of_parts = len(partition)
    return partition

'''
Assigns partitions to nodes uniformly so that the size of all partition is approximately the same
'''
def uniform_partition(graph, nb_parts=default_numbers_of_parts):
    
    nb = len(graph.nodes)
    segment = nb//nb_parts
    loners = nb % nb_parts
    partitions = {i:[j for j in range(i*segment, (i+1)*segment)] for i in range(nb_parts)}
    if loners != 0:
        for i in range(loners):
            partitions[i].append(nb-1-i)

    return partitions


partitions = [random_partition, greedy_modularity_community_partition, louvain_communities_community_partition, degree_partition, pagerank_clustering_partition, uniform_partition]