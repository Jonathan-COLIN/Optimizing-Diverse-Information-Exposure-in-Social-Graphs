import numpy as np
import scipy.sparse as ss
from scipy.special import softmax
import networkx as nx
import pulp
import random
import math
import multiprocessing as mp

from Functions.Matrix import *


def find_closest_candidates_different_home(graph, u, Prow, amount):
    A = nx.adjacency_matrix(graph)
    t = ss.lil_array(A.shape[0])
    target = ss.lil_matrix(shape=(A.shape[0],1), dtype=np.float64)
    target[u,0] = 1 
    target = target.tocsr()
    d = 1
    while len(t[t >0]) < amount and d<10:
        temp = A
        for _ in range(d):
            temp @= A
        temp = (temp @ target).T
        for node in nx.neighbors(graph,u):
            vecter = ss.lil_array(shape=(A.shape[0],1))
            vecter[node,0] = 1 
            vecter = vecter.tocsr()
            temp += (A @ vecter).T
        d += 1
        t = np.multiply(temp, Prow).todense()[0]
    
    return t

def selectPartition(distr, strategy):
    c = [j for j in range(len(distr)) if distr[j] < 0]
    targetParts = sorted(c, reverse=True)
    if strategy =='RMPB':
        np.random.shuffle(targetParts)
    if strategy == 'BMPB':
        p = distribution([distr[j] for j in c])
        targetParts = np.random.choice(c,replace=False, size=len(c),p=softmax(100*p.flatten()))
    
    return targetParts

def selectCandidates(graph, u, PMatrix, targetPart, strategy, k):
    A = nx.adjacency_matrix(graph)
    Allcandidates = (PMatrix @ A)[targetPart,:].toarray()[0]
    Allcandidates = [i for i in range(len(Allcandidates)) if Allcandidates[i] > 0]
    number = np.min([len(Allcandidates), k])
    if strategy == 'all':
        candidates = Allcandidates
    if strategy == 'ssbr':
        candidates = np.random.choice(Allcandidates, number, replace=False)
    if strategy == 'ssdb':
        p = [nx.degree(graph,node) for node in Allcandidates]
        p = distribution(p)
        candidates =  np.random.choice(Allcandidates, size=number, replace=False, p=p)
    if strategy == 'sstb':
        candace = find_closest_candidates_different_home(graph, u, PMatrix[targetPart], number)
        candace = [i for i in range(len(candace)) if candace[i] > 0]
        if len(candace) != 0:
            candidates = candace
        else:
            candidates = Allcandidates

    return np.array(candidates, dtype=int)

def selectTargets(graph, k):
    degreedict = {}

    for node,degree in dict(nx.degree(graph)).items():
        if degree in degreedict:
            degreedict[degree].append(node)
        else:
            degreedict[degree] = [node]

    lopdict = {'ahigh':[], 'bmedium':[], 'clow':[]}

    levels = list(lopdict.keys())
    i = 0
    templist = []
    for degree, nodelist in degreedict.items():

        for node in nodelist:
            templist.append(node)

            if len(templist) > len(graph.nodes)/3:
                lopdict[levels[i]] = templist
                i += 1
                templist = []

        lopdict[levels[i]] = templist

    # in the following loop we pick turn by turn the first node of each list
    # we want the highest degree nodes which is already at the start of the ['ahigh'] list so no changes
    # we want nodes with medium degrees to be uniformly picked so we shuffle the ['bmedium'] list
    # we want the lowest degree nodes so we have to invert the ['clow'] list
    np.random.shuffle(lopdict['bmedium'])
    lopdict['clow'] = lopdict['clow'][::-1]

    loop = [b for a,b in sorted([[degree,nodelist] for degree, nodelist in lopdict.items()], key=lambda x : x[0], reverse=False)]

    res = []
    i = 0
    while len(res) < k:
        if i // len(loop) < len(loop[i % len(loop)]):
            res.append(loop[i % len(loop)][i // len(loop)])
        i += 1
    
    return res

##########################################################################################
### NEURIPS ##############################################################################
##########################################################################################

def Disagreement(graph, z):
    return np.sum([(z[i] - z[j])**2 for i,j in graph.edges])

def Polarization(z):
    return z.T @ z

def Indexz(graph, z):
    return Disagreement(graph,z) + Polarization(z)

def Indexs(omega, s):
    return (s.T @ omega @ s).todense()[0,0]

def Neurips(graph, target, k, s=None):
    I = ss.identity(graph.number_of_nodes())
    B = nx.incidence_matrix(graph).T
    L = B.T @ B

    OMEG = ss.linalg.inv((I + L).tocsc()).tocsc()

    if s == None:
        s = np.random.normal(scale=1, size=(1,len(graph.nodes))).T
        s = ss.csc_array((s - np.mean(s))/np.max(np.abs(s)))
    

    def bij(i,j, g=graph):
        indicatrice = ss.lil_array((graph.number_of_nodes(),1),dtype=float)
        indicatrice[i] = 1
        indicatrice[j] = 1
        indicatrice = indicatrice.tocsr().T
        return indicatrice

    def rij(i,j, O):
        b = bij(i,j)
        return (b @ O @ b.T).todense()[0][0]

    def zu(u, O):
        indicatrice = ss.lil_array((O.shape[0],1),dtype=float)
        indicatrice[u] = 1
        indicatrice = indicatrice.tocsr().T
        return (indicatrice @ O @ s).todense()[0,0]
    
    result = []
    #init = Indexs(OMEGA,s)
    candidate_edges = [[target,node] for node in graph.nodes() if [target,node] not in graph.edges()]
    for i in range(k):
        best_edge = [target,target]
        best_increase = -np.inf
        for a,b in candidate_edges:
            edge = [a,b]
            if edge not in result and edge not in graph.edges:
                Omeganext = increment_matrix_inverse(OMEG,edge)

                f = (zu(edge[0], Omeganext) - zu(edge[1], Omeganext))/(1 + rij(edge[0],edge[1], Omeganext))

                if f > best_increase:
                    best_edge = edge
                    best_increase = f
        
        result.append([best_edge[0],best_edge[1]])
        OMEG = increment_matrix_inverse(OMEG,best_edge).tocsc()

    return [node for _,node in result]


##########################################################################################
### DEMAINE ##############################################################################
##########################################################################################

def GlobalDemaine(graph, target, k, x=None, sigma=1):
    if target == None:
        S = [np.random.choice(graph.nodes,1)[0]]
    else:
        S = [target]
    
    if x == None:
        x = 0.75 * nx.eccentricity(graph,target)

    while x > 0.5:

        for node in graph.nodes:
            t = [nx.shortest_path_length(graph, node, u) for u in S]

            if all([tt >= 2*x for tt in t ]):
                S.append(node)
        
        if len(S) >= k+1:
            break
        else:
            #S = [target]
            x = 0.75 * x
        
    return S[1:k+1]

def LINEARPROGRAM(graph,source, k,epsilon=0.1):

    r = nx.eccentricity(graph,0)
    i_range = range(0,int(math.log((r),(1+epsilon))))
    X_range = [2] + [int(np.ceil(r/((1+epsilon)**i_range[-i]))) for i in range(1,len(i_range))]

    for X in list(set(X_range)):
        # Create LP problem
        prob = pulp.LpProblem("k-median-clustering", pulp.LpMinimize)

        xij = {node:[] for node in graph.nodes() if node != source and nx.shortest_path_length(graph,source,node) > X}
        for key in xij.keys():
            xij[key] = [node for node in nx.bfs_tree(graph,source=key,depth_limit=X).nodes() if node in xij.keys()]

        # Decision variables
        x = pulp.LpVariable.dicts("x", [(i, j) for i in xij.keys() for j in xij[i]], 0, 1)
        y = pulp.LpVariable.dicts("y", xij.keys(), 0, 1)
        
        # Objective function
        prob += pulp.lpSum(x[i, j] for i in xij.keys() for j in xij[i])
        
        # Constraints
        for i in xij.keys():
            prob += pulp.lpSum(x[i, j] for j in xij[i]) == 1
            for j in xij[i]:
                prob += pulp.LpConstraint(y[i] - x[i, j], sense=pulp.LpConstraintGE)
        
        prob += pulp.lpSum(y[i] for i in xij) <= k
        
        # Solve LP
        try:
            prob.solve(pulp.apis.PULP_CBC_CMD(msg=False))
        except:
            pass
        
        if prob.status == 1:
            break

    return {node: pulp.value(y[node]) for node in xij.keys()}, X

def SingleSourceDemaine(graph,target,k):

    recommendations = []
    
    COPY = graph.copy()  

    for _ in range(k):
        candidates, _ = LINEARPROGRAM(COPY,target,k)
        if candidates == {}:
            potentials = [node for node in graph.nodes() if [target,node] not in graph.edges()]
            random.shuffle(potentials)
            candidates = {node:1/len(potentials) for node in potentials}
        for node,p in candidates.items():
            if random.random() < p and len(recommendations) < k:
                recommendations.append(node)
        
        COPY.add_edges_from([[target,node] for node in recommendations if [target,node] not in graph.edges])

        if len(recommendations) >= k:
            break
    
    return recommendations

##########################################################################################
### Simple Recommenders ##################################################################
##########################################################################################

def common_neighbors_recommendation(graph, target, k):
    already = list(nx.neighbors(graph,target))
    setarget = set(graph.neighbors(target))
    results = []

    results = np.array([[node, len(setarget.intersection(set(graph.neighbors(node))))] for node in graph.nodes if node not in already])
    results = sorted(results, key=lambda x: x[1])

    if k < len(results):
        return [a for a,_ in results[:k]]
    else:
        for _ in range(len(results),k):
            results.append(results[-1])
        return [a for a,_ in results]

def jacc_recommendation(graph, target, k):
    already = list(nx.neighbors(graph,target))
    setarget = set(graph.neighbors(target))
    results = []
    for _ in range(k):
        best_node = target
        best_score = -1
        for node in graph.nodes():
            if node != target and node not in results and node not in already:

                try:
                    score = len(setarget.intersection(set(graph.neighbors(node)))) / len(setarget.union(set(graph.neighbors(node))))
                except:
                    score = 0

                if score > best_score:
                    best_node = node
                    best_score = score
        results.append(best_node)
    
    return results

def pref_recommendation(graph,target,k):
    already = list(nx.neighbors(graph,target))
    results = []
    for _ in range(k):
        best_node = target
        best_score = -1
        for node in graph.nodes():
            if node != target and node not in results and node not in already:

                score = nx.degree(graph,node)

                if score > best_score:
                    best_node = node
                    best_score = score
        results.append(best_node)
    
    return results

def invDist_recommendation(graph,target,k):
    already = list(nx.neighbors(graph,target))
    results = []
    for _ in range(k):
        best_node = target
        best_score = -1
        for node in graph.nodes():
            if node != target and node not in results and node not in already:

                score = 0
                try:
                    score = 1/nx.shortest_path_length(graph,target,node)
                except:
                    score = 0

                if score > best_score:
                    best_node = node
                    best_score = score
        results.append(best_node)
    
    return results

def random_recommendation(graph, target, k):
    recommendations = []
    neighbours = list(nx.neighbors(graph,target))
    while len(recommendations) < k:
        node = np.random.randint(0,len(graph.nodes))
        if node not in neighbours and node not in recommendations and node != target:
            recommendations.append(node)
        if len(recommendations) > k:
            break
        if len(recommendations) + len(neighbours) >= len(graph.nodes()):
            break
    
    return recommendations


##########################################################################################
### Small World ##########################################################################
##########################################################################################

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

def edgeEffect_recommendation(graph,target,k):
    already = list(nx.neighbors(graph,target))
    results = []
    for _ in range(k):
        best_node = target
        best_score = -1
        for node in graph.nodes():
            if node != target and node not in results and node not in already:

                score = EdgeEffect(graph,target,node)

                if score > best_score:
                    best_node = node
                    best_score = score
        results.append(best_node)
    
    return results

def effectEstimation_recommendation(graph,target,k):
    already = list(nx.neighbors(graph,target))
    results = []
    for _ in range(k):
        best_node = target
        best_score = -1
        for node in graph.nodes():
            if node != target and node not in results and node not in already:

                score = EffectEstimation(graph, [target, node])

                if score > best_score:
                    best_node = node
                    best_score = score
        results.append(best_node)
    
    return results


##########################################################################################
### MAIN    ##############################################################################
##########################################################################################

def recommendMain(graph, u, k, Wfunction, PMatrix, candidates=None, strategy='greedy', targetQ=None):

    if candidates == None:
        candidates = list(graph.nodes())
    if targetQ is None:
        targetQ = (np.ones(PMatrix.shape[0])/PMatrix.shape[0]).reshape((PMatrix.shape[0],1))

    def get_one(Wf, Adj, P, u,target, candidates, targetQ, queue, invP=None):
        best_candidate = u
        best_score = np.inf
        for candidate in candidates:
            if Adj[u, candidate] == 0 and candidate != u:

                objective, _ = OBJECTIVE(P, Wf(Adj, P, u, targetQ, invP), target, targetQ)

                if objective < best_score:
                    best_candidate = candidate
                    best_score = objective
        queue.put([best_candidate, best_score])
        

    num_processes = mp.cpu_count()

    Ag = nx.adjacency_matrix(graph)
    Ag = rowstochastic(Ag).tolil()
    P = PMatrix
    
    target = ss.lil_array((Ag.shape[0],1),dtype=float)
    target[u] = 1
    target = target.tocsr()

    recommendations = []
    scores = []
    if strategy == 'greedy':
        chunk_size = len(candidates) // num_processes
        candidate_chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        for i in range(k):
            result_queue = mp.Queue()
            processes = []
            if 'WPPR' in Wfunction.__name__ or 'WFJ' in Wfunction.__name__:
                invP = pseudo_inverse(P)
            for chunk in candidate_chunks:
                if 'WPPR' in Wfunction.__name__ or 'WFJ' in Wfunction.__name__:
                    p = mp.Process(target=get_one, args=(Wfunction,Ag,P,u,target,chunk,targetQ, result_queue, invP))
                else:
                    p = mp.Process(target=get_one, args=(Wfunction,Ag,P,u,target,chunk,targetQ, result_queue))
                processes.append(p)
                p.start()

            # Wait for all processes to finish
            for p in processes:
                p.join()

            results = [result_queue.get() for _ in range(len(candidate_chunks))]
            results = sorted(results, key=lambda x: x[1], reverse=False)
            recommendations.append(results[0][0])
            scores.append(results[0][1])
            Ag[u, results[0][0]] = 1/(np.sum(Ag[[u],:])+1)

        return recommendations, scores
    
    if 'PB' in strategy:
        partition_strategy, candidate_strategy = strategy.split('-')

        baseW = Wfunction(Ag, P, u, targetQ)
        _, disitr = OBJECTIVE(P,baseW,target,targetQ)
        sorted_candidate_partition = selectPartition(disitr, partition_strategy)
        candidates = np.array([],dtype=int)
        for targetPart in sorted_candidate_partition:
            candidates = np.concatenate([candidates, selectCandidates(graph, u, P, targetPart, candidate_strategy, k*10)], dtype=int)
            if len(candidates) > k:
                break

        chunk_size = np.max([len(candidates) // num_processes, 1])
        candidate_chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

        for _ in range(k):
    
            result_queue = mp.Queue()
            processes = []
            if 'WPPR' in Wfunction.__name__ or 'WFJ' in Wfunction.__name__:
                invP = pseudo_inverse(P)
            for chunk in candidate_chunks:
                if 'WPPR' in Wfunction.__name__ or 'WFJ' in Wfunction.__name__:
                    p = mp.Process(target=get_one, args=(Wfunction,Ag,P,u,target,chunk,targetQ, result_queue, invP))
                else:
                    p = mp.Process(target=get_one, args=(Wfunction,Ag,P,u,target,chunk,targetQ, result_queue))
                processes.append(p)
                p.start()

            # Wait for all processes to finish
            for p in processes:
                p.join()

            results = sorted([result_queue.get() for _ in range(len(candidate_chunks))],key=lambda x: x[1], reverse=False)
            recommendations.append(results[0][0])
            Ag[results[0][0], u] = 1/(np.sum(Ag[[u],:])+1)
            scores.append(results[0][1])

        return recommendations, scores
    
    if strategy == 'convexity':
        COPY = graph.copy()
        invP = None
        if 'WPPR' in Wfunction.__name__ or 'WFJ' in Wfunction.__name__:
            invP = pseudo_inverse(P)
        for _ in range(k):
            recommendations.append(GD_recommendation(COPY,u, PMatrix, targetQ, Wfunction, invP))
            COPY.add_edge(u,recommendations[-1])

        return recommendations, OBJECTIVE_list(graph, PMatrix, Wfunction, u, targetQ, recommendations)

    return None


models = [recommendMain, GlobalDemaine, SingleSourceDemaine, Neurips, common_neighbors_recommendation, random_recommendation]