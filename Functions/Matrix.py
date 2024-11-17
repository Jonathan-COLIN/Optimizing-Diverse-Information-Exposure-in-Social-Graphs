import numpy as np
import networkx as nx
import scipy.sparse as ss
from scipy.special import softmax


""" Given a graph and a partition dictionary {parts:[nodes]} returns the P matrix nb parts x nb nodes"""
def partition_matrix(graph, partition):
    P = ss.lil_matrix((len(partition.keys()), len(graph.nodes)))
    for i, nodelist in partition.items():
        for node in nodelist:
            P[i,node] = 1
    return ss.csr_matrix(P,dtype=np.float64)

def continuous_P(graph, attr_name):
    Pnparray = []
    for node in graph.nodes():
        Pnparray.append(graph.nodes[node][attr_name])
    return ss.csr_matrix(Pnparray).T

def produce_partition(graph, partitioname, c_attr = None):
    if partitioname == 'louvain_communities_community_partition':
        from Functions.Partitions import louvain_communities_community_partition
        partition = louvain_communities_community_partition(graph)
        P = partition_matrix(graph, partition)
    elif partitioname == 'ground-truth':
        partition = {}
        for node in graph.nodes(data=True):
            if node[1]['ground-truth'] in partition.keys():
                partition[node[1]['ground-truth']].append(node[0])
            else:
                partition[node[1]['ground-truth']] = [node[0]]
        P = partition_matrix(graph, partition)
    elif 'continuous' in partitioname:
        partition = {}
        P = continuous_P(graph, attr_name=c_attr)
    
    return partition, P

def Matrix_edge_addition(A,u,v):
    temp = A.tolil()
    temp[u,v] = 1
    temp[v,u] = 1
    return temp.tocsr()

"""given an array of numbers, returns the distribution"""
def distribution(arr):
    total = np.sum(arr)
    if total == 0:
        return arr
    else:
        return arr/total

def uniformise(matrix):
    return matrix/np.sum(matrix, axis=1)

"""Objective Distribution Functions"""
def UniformQ(PMatrix, WMatrix, u):
    return (np.ones(PMatrix.shape[0])/PMatrix.shape[0]).reshape((PMatrix.shape[0],1))

def PreferentialQ(PMatrix, WMatrix, u):
    target = ss.lil_matrix((WMatrix.shape[0],1))
    target[u,0] = 1 # np.random.randint(len(graph.nodes))
    target = ss.csr_array(target,dtype=np.float64)
    targetQ = UniformQ(PMatrix, WMatrix, u)
    home = np.argmax(PMatrix @ target)

    if WMatrix.shape[0] == WMatrix.shape[1]:
        Wu = WMatrix @ target
    else:
        Wu = WMatrix

    base = np.abs((PMatrix @ Wu).multiply(targetQ).T.toarray().flatten())
    harmonizer = np.array([1 + 1/PMatrix.shape[0] * np.exp(-((x - home) ** 2) / (2 * (PMatrix.shape[0]/2) ** 2)) if x != home else 1 for x in range(PMatrix.shape[0])])
    baseliner = [base[home]/10 if i != home else 0 for i in range(PMatrix.shape[0])]
    return distribution((base + baseliner) * harmonizer).reshape(PMatrix.shape[0],1)

"""It returns zero if the input is less than zero otherwise it returns the given input"""
def RELU(x):
    x1=np.zeros(shape=x.shape)
    for i in range(len(x)):
        if x[i]<0:
            x1[i] = -x[i]
        else:
            x1[i] = 0

    return x1

"""Given a square matrix, returns the corresponding matrix where each line sums to 1"""
def rowstochastic(A):
    return A.multiply(1/A.sum(axis=1))

"""W functions"""
def WBFS(A,u, d=3):
    A2 = A@A
    return A + A2 + A2@A

def WPPR(A,u, alpha=0.05):
    I = ss.eye(A.shape[0])
    #return (1- alpha) * ss.linalg.inv((ss.eye(A.shape[0]) - A.multiply(alpha)).tocsc())
    #return (1 - alpha) * ss.linalg.spsolve((ss.eye(A.shape[0])-A.multiply(alpha)).tocsc(),ss.eye(A.shape[0]).tocsc())
    #return (1 - alpha) * ss.linalg.bicg((I - A.multiply(alpha)).tocsc(), I)
    return ss.csr_matrix((1- alpha) * np.linalg.inv((I - A.multiply(alpha)).todense()))

def WFJ(A,u, alpha=0.05):
    #return (1- alpha) * ss.linalg.inv((ss.eye(A.shape[0]) - (A.T).multiply(alpha)).tocsc())
    #return (1 - alpha) * ss.linalg.spsolve((ss.eye(A.shape[0])-(A.T).multiply(alpha)).tocsc(),ss.eye(A.shape[0]).tocsc())
    return ss.csr_matrix((1- alpha) * np.linalg.inv((ss.eye(A.shape[0]) - (A.T).multiply(alpha)).todense()))

"""Incremental Wfunctions"""
def increment_matrix_inverse(A,edge):
    c = ss.lil_array((A.shape[0],1),dtype=float)
    d = ss.lil_array((A.shape[0],1),dtype=float)
    c[edge[0]] = 1
    d[edge[1]] = 1
    c = c.tocsr()
    d = d.tocsr().T

    beta = 1 + (d @ A @ c).todense()[0][0]

    return A - (1/beta) * (A @ c @ d @ A)

def increment_WBFS(A, edge, oldW):
    u,v = edge
    C = ss.lil_matrix(A.shape, dtype=int)
    C[u,v] = 1
    C[:, [v]] += A[[u], :].T
    C[[u], :] += A[:, [v]].T
    uv2 = np.sum(A[(A[[u],:] == 1).toarray().flatten()], axis=0)
    vv2 = np.sum(A[(A[:,[v]] == 1).toarray().flatten()], axis=0)
    C[[u],:] += vv2.reshape(C[[u],:].shape)
    C[:,[v]] += uv2.reshape(C[:,[v]].shape)
    check = np.sum(A[[u],:]==1)
    if check > 0:
        C[(A[[u],:]==1).toarray().flatten(),:] += ss.vstack([A[:, [v]].T for _ in range(np.sum(A[[u],:]==1))])

    return oldW + C

def increment_WPPR(A, edge, alpha = 0.05):
    matrix = (ss.eye(A.shape[0]) - A.multiply(alpha))
    return (1-alpha) * increment_matrix_inverse(matrix,edge)

def increment_WFJ(A, edge, alpha = 0.05):
    matrix = (ss.eye(A.shape[0]) - (A.T).multiply(alpha))
    return (1-alpha) * increment_matrix_inverse(matrix, edge)

"""BICG Wfunctions"""
def pseudo_inverse(P):
    m = ss.csr_matrix(P,dtype=np.float64)
    try:
        if max(P.shape) < 1e6:
            up, s, vt = ss.linalg.svds(m, k=min(P.shape)-1, solver='arpack', tol=1e-2)
            s_inv = ss.diags(1 / s)
            Pinv = ss.csr_matrix(vt.T @ s_inv @ up.T)
        else:
            Pinv = P.T

    except:
        Pinv = P.T
        
    return Pinv


def WBFS_bicg(A, P, node, targetQ, invP=None):
    u = ss.lil_matrix((A.shape[0],1))
    u[node] = 1

    return (A + ss.linalg.matrix_power(A,2)) @ u.tocsr()

def WPPR_bicg(A, P, node, targetQ, invP=None, alpha=0.05):
    u = ss.lil_array((A.shape[0],1), dtype=np.float64)
    u[node] = 1
    u = u.tocsr()
    I = ss.eye(A.shape[0])
    B = (I- alpha*A)
    Pinv = pseudo_inverse(P) if invP is None else invP
    rhs = (Pinv @ targetQ)
    res, exitcode = ss.linalg.bicgstab(B, u.todense() + rhs, atol=1e-4, rtol=1e-4, maxiter=int(5e3))
    if exitcode >= 0:
        res = ss.csr_matrix(res)
    else:
        res = ss.csr_matrix(A[[node],:]) + uniformise(ss.random(1, A.shape[0], density=0.1, dtype=np.float64))
    return res.reshape((A.shape[0],1)).tocsr()

def WFJ_bicg(A, P, node, targetQ, invP=None, alpha=0.05):
    u = ss.lil_array((A.shape[0],1), dtype=np.float64)
    u[node] = 1
    u = u.tocsr()
    I = ss.eye(A.shape[0])
    B = (I- alpha*A.T)
    Pinv = pseudo_inverse(P) if invP is None else invP
    rhs = (Pinv @ targetQ)
    res, exitcode = ss.linalg.bicgstab(B, u.todense() + rhs, atol=1e-5, rtol=1e-5, maxiter=int(5e3))
    if exitcode >= 0:
        res = ss.csr_matrix(res)
    else:
        res = ss.csr_matrix(A[[node],:]) + uniformise(ss.random(1, A.shape[0], density=0.1, dtype=np.float64))
    return res.reshape((A.shape[0],1)).tocsr()


""" Given 
    P partition matrix
    W matrix of transition
    targetarray indicator array of target node
    endQ the goal distribution
    returns the corresponding value of the objective functions
    """
def OBJECTIVE(PMatrix, W, targetarray, endQ):
    
    if W.shape[0] == W.shape[1]:
        disitr = distribution(((PMatrix @ W) @ targetarray).todense()) - endQ
    elif W.shape[0] > W.shape[1] and W.shape[1] == 1:
        disitr = distribution(PMatrix @ W) - endQ

    return np.linalg.norm(disitr, ord=2), disitr

def OBJECTIVE_list(graph, PMatrix, Wfunction, u, targetQ, l):

    m = nx.adjacency_matrix(graph)
    m = rowstochastic(m).tolil()
    res = []

    for node in l:
        m = Matrix_edge_addition(m,u,node)
        dob = Wfunction(m, PMatrix, u, targetQ)
        res.append(OBJECTIVE(PMatrix, dob, u, targetQ)[0])
    return res



""" Gradient Descent Algorithm
    Executes one step of the descent
    which corresponds to one recommendation"""
def GD_recommendation(graph, target_node, PMatrix, objectiveQ, WFunction, invP=None, alpha=0.2, choice='max', mu=0.2):
    Adj = nx.adjacency_matrix(graph)
    Adj = rowstochastic(Adj).tolil()
    n = Adj.shape[0]

    target = ss.lil_matrix((Adj.shape[0],1))
    target[target_node,0] = 1
    target = ss.csr_array(target,dtype=np.float64)

    baseW = WFunction(Adj, PMatrix, target_node, objectiveQ)
    d = RELU((distribution(PMatrix @ baseW) - objectiveQ)).T

    if 'BFS' in WFunction.__name__:
        Adj2 = ss.csr_matrix.power(Adj.tocsr(),2)
        dW = - ss.eye(n) - Adj - ss.eye(n).multiply(Adj2.T.tolil()[target_node,target_node]) - Adj2
        dOu = 2*d @ PMatrix @ dW
        newA = ss.csr_array(arg1=(Adj[[target_node], :] - mu * dOu)[0].reshape((Adj.shape[0],1)))
    if 'PPR' in WFunction.__name__:
        Qinvu = WFunction(Adj, PMatrix, target_node, objectiveQ, invP)
        dWu = - alpha * (1 - alpha) * Qinvu.multiply(Qinvu.tolil()[target_node]).T
        dOu = ss.csr_matrix(2 * dWu.multiply(d @ PMatrix))
        newA = (Adj[[target_node], :] - mu * dOu).T
    if 'FJ' in WFunction.__name__:
        Qinvu = WFunction(Adj, PMatrix, target_node, objectiveQ, invP)
        dWu = -alpha * (1 - alpha) * Qinvu.multiply(Qinvu.tolil()[target_node]).T
        dOu = 2 *dWu.multiply(d @ PMatrix)
        newA = (Adj[[target_node], :] - mu * dOu).T
    
    
    #projection
    newA = newA.tolil()
    newA[newA < 0] = 0
    newA[target_node] = 0
    newA[Adj[[target_node],:].todense()[0] > 0] = 0
    newA = newA.tocsr()

    if choice == 'max':
        return newA.argmax()
    if choice == 'random':
        return np.random.choice(range(len(newA)),size=1,replace=False,p=softmax(50*newA))[0]





