import numpy as np
import networkx as nx
from random import random


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def EdgeEffect(graph,u,v):

  x = u
  y = v
  R = 0

  Ax = [ u for u in graph.nodes() if nx.shortest_path_length(graph, source = u, target = y) + 1 < nx.shortest_path_length(graph, source = u, target = x)]
  Ay = [ u for u in graph.nodes() if nx.shortest_path_length(graph, source = u, target = x) + 1 < nx.shortest_path_length(graph, source = u, target = y)]

  for u in Ax:

    for v in graph.nodes():
      new = nx.shortest_path_length(graph, source = v, target = x) + nx.shortest_path_length(graph, source = u, target = y) + 1
      old = nx.shortest_path_length(graph, source = u, target = v)

      if new < old:
        R += old - new
  return R

def EffectEstimation(graph, edge):
  x = edge[0]
  y = edge[1]
  R = graph

  dold = nx.shortest_path_length(graph, source = x, target = y)

  Tx = nx.bfs_tree(graph, x, depth_limit=dold)
  Ty = nx.bfs_tree(graph, y, depth_limit=dold)

  Ax_d = { i:[u for u in Tx if ((nx.shortest_path_length(graph, source = u, target = y) + 1 - nx.shortest_path_length(graph, source = u, target = x))) == -i] for i in range(1,dold)}

  Ay_d = { i:[u for u in Ty if (nx.shortest_path_length(graph, source = u, target = x) + 1 - nx.shortest_path_length(graph, source = u, target = y)) == -i] for i in range(1,dold)}

  R = 0
  if dold > 2:
    for i in range(1,dold):
      for j in range(dold-i,dold):
        R += len(Ax_d[i])*len(Ay_d[j])*np.min([i,j])
  if dold == 2:
    Xneighbors = [n for n in graph.neighbors(x)]
    Yneighbors = [n for n in graph.neighbors(y)]
    R += len(np.intersect1d(Xneighbors, Ay_d)) * len(np.intersect1d(Yneighbors, Ax_d)) 
  
  return R

'''
Biased personalized pagerank with monte carlo estimation
'''
def PR_perso_mc(G,i,j, walks = 10, alpha = 0.2):

  found = 0
  for _ in range(walks):
    next = i
    while(random.random() < 1-alpha):
      try:
        surroundings = list(nx.neighbors(G,next))
        probs = [2+G.degree(n)/2 if G.nodes[n]['partition'] == G.nodes[next]['partition'] else 1+G.degree(n)/2 for n in surroundings]
        total = np.sum(probs)
        next = np.random.choice(surroundings,size=1,p=[d/total for d in probs])[0]
        if next == j:
          found += 1
      except:
        next = i
  return found/walks

'''
Unbiased personalized pagerank with monte carlo estimation
'''
def mc_pr(graph, root_node, target_node, num_samples=100, alpha=0.15):
    found = 0

    current_node = root_node
    for _ in range(num_samples):
        if random.random() < alpha:
            current_node = root_node
        else:
            neighbors = list(graph.neighbors(current_node))
            if len(neighbors) == 0:
                current_node = root_node
            else:
                current_node = random.choice(neighbors)
        if current_node == target_node:
           found += 1

    return found / num_samples

def common_neighbors(G,u,v):
    return len(set(G.neighbors(u)).intersection(set(G.neighbors(v))))

'''
Jaccard similarity
'''
def jacc(G,i,j):
  try:
    res = len(set(G.neighbors(i)).intersection(set(G.neighbors(j)))) / len(set(G.neighbors(i)).union(set(G.neighbors(j))))
  except:
    res = 0
  return res

'''
Preferential attachment
'''
def pref(G,i,j):
  return nx.degree(G, i) * nx.degree(G,j)

'''
Inverse Distance
'''
def inv_dist(G,i,j):
  res = 0
  try:
    res = 1/nx.shortest_path_length(G,i,j)
  except:
    res = 0
  return res
