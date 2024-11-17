import numpy as np
import networkx as nx

log = np.log
serendipity = 0.005


'''
Random Walks rooted at u biased towards nodes with higher degrees and nodes from the same partition
Returns a dictionary keys are partitions, values are the number of times the key has been visited during the random walks
'''
def biased_PPR_vision(G,node, nb_walks = 100, walk_length=5, go_back_home = 0.1):
  global number_of_parts
  vision = {k:serendipity for k in range(number_of_parts)}

  for _ in range(nb_walks):
    step = 0
    current_node = node
    while step < walk_length:

      surroundings = list(nx.neighbors(G,current_node))
      probs = [G.degree(n) if G.nodes[node]['partition'] == G.nodes[n]['partition'] else G.degree(n)/2 for n in surroundings]
      total = np.sum(probs)
      try:
        next = np.random.choice(surroundings,size=1,p=[d/total for d in probs])[0] #
      except:
        if current_node == node:
          return vision

      if np.random.random() < go_back_home:
        if next in vision:
          vision[G.nodes[next]['partition']] += 1
        else:
          vision[G.nodes[next]['partition']] = 1
        break
      
      if next in vision:
        vision[G.nodes[node]['partition']] += 1
      else:
        vision[G.nodes[node]['partition']] = 1
      step = step + 1
      current_node = next
  
  return vision

'''
Unbiased random walks rooted at u
Returns a dictionary keys are partitions, values are the number of times the key has been visited during the random walks
'''
def PPR_vision(G,node, nb_walks = 5, walk_length=4, go_back_home = 0.1):
  vision = {k:serendipity for k in range(number_of_parts)}

  for _ in range(nb_walks):
    step = 0
    current_node = node
    while step < walk_length:

      surroundings = list(nx.neighbors(G,current_node))
      probs = [1/len(surroundings) for node in surroundings]
      try:
        next = np.random.choice(surroundings,size=1,p=probs)[0]
      except:
        if current_node == node:
          return vision

      if np.random.random() < go_back_home:
        if next in vision:
          vision[next] = vision[next] + 1
        else:
          vision[next] = 1
        break
      
      if next in vision:
        vision[next] = vision[next] + 1
      else:
        vision[next] = 1
      step = step + 1
      current_node = next
  
  return vision

'''
Given a Dictionary produced by a Random walk vision
Produces a distribution of probabilities for each partition
'''
def PPRVISION2DISTR(graph,vision):
  total = np.sum(list(vision.values()))
  return { k: v/total if v != 0 else serendipity for k,v in vision.items()}


'''
A single random walk for a given node
'''
def walk_segment(graph,source, walk_length=10, go_back_home = 0.15):
  step = 0
  current_node = source
  global number_of_parts
  walk = {k:0 for k in range(number_of_parts)}
  while step < walk_length:
    step += 1
    surroundings = list(nx.neighbors(graph,current_node))
    probs = [graph.degree(node) for node in surroundings]
    total = np.sum(probs)
    next = np.random.choice(surroundings, size=1, p=[d/total for d in probs])[0]

    if np.random.random() < go_back_home and len(walk) > 0:
      next = source
    else:
      walk[graph.nodes[next]['partition']] += 1
    current_node = next
  return walk

'''
In order to do fastPPR, this functions stacks to given walks together
given 2 dictionaries, for each key, sums their values
'''
def PPR_stack(vision, to_stack):
  for k,v in to_stack.items():
      if k in vision.keys():
        vision[k] += v
      else:
        vision[k] = v
  return vision

'''
Old version of the BFS version
kept for history tracking
'''
def BFS_vision_very_old(G,node):

  def BFS(G,viewed,distribution,distance,deprecated=[]):
    if distance == 1:
      home = G.nodes[viewed[0]]['partition']
    else:
      home = G.nodes[deprecated[0]]['partition']
    new_view = set()

    for node in viewed:
      if not node in deprecated:
        for neigh in nx.neighbors(G,node):

          if not neigh in viewed:
            current = G.nodes[neigh]['partition']
            if current != home:
              if not current in distribution:
                distribution[current] = [distance]
              else:
                distribution[current].append(distance)
              pass
            else: #if current == home
              new_view.add(neigh)
          else: #if node in viewed
            pass

    if len(new_view) == 0:
      return
    for node in viewed:
      deprecated.append(node)
    BFS(G,[node for node in new_view if not node in deprecated],distribution,distance+1,deprecated)
  
  distribution = {}

  BFS(G,[node],distribution,1)

  return distribution

'''
Old version of the BFS version
kept for history tracking
'''
def BFS_vision_old(graph,source):
  global number_of_parts
  tree = nx.bfs_tree(graph,source)

  home = graph.nodes[source]['partition']
  visiting = [source]
  distribution = {k:[] for k in range(number_of_parts)}
  depth = 0

  while visiting != []:
    visitang = []
    depth += 1
    for node in visiting:
        guest = graph.nodes[node]['partition']
        if guest == home:
            if home in distribution:
                distribution[home].append(depth)
            else:
                distribution[home] = [depth]
            for nei in tree.neighbors(node):
                visitang.append(nei)
        else:
            if guest in distribution:
                distribution[guest].append(depth)
            else:
                distribution[guest] = [depth]
    visiting = visitang
  
  return distribution

'''
Old version, kept for tracking
'''
def deprecated_BFSVISION2DISTR(G,vision):
    a,b = np.unique([v for k,values in vision.items() for v in values],return_counts=True)
    powertwo = 2**(len(a)-1)
    diviser = np.sum([(b[i]*powertwo)/(2**(i-1)) for i in range(len(a))])
    weight = powertwo/diviser
    translator = {a[i]:weight/(2**(i-1)) for i in range(len(a))}
    return { k:np.sum([translator[v] for v in values]) for k,values in vision.items()}

'''
Builds the BFS tree of a node in a graph 
with the added condition of stopping when changing partition.
'''
def BFS_TREE(G,root,resultGraph):
  home = G.nodes[root]['partition']
  visited = set()
  currentLevel = [root]
  while currentLevel:
    for v in currentLevel:
      visited.add(v)
    nextLevel = set()
    levelGraph = {v:set() for v in currentLevel}
    for v in currentLevel:
      for w in G[v]:
        if w not in visited:
          if home == G.nodes[w]['partition']:
            levelGraph[v].add(w)
            nextLevel.add(w)
            visited.add(w)
          else:
            levelGraph[v].add(w)
            visited.add(w)
    yield levelGraph
    temp = nx.DiGraph(levelGraph)
    resultGraph.add_edges_from(temp.edges())
    currentLevel = nextLevel

'''
Builds the directed graph corresponding to the BFS tree of a node
'''
def BFS_graph(G,root):
  res = nx.DiGraph()

  for element in BFS_TREE(G,root,res):
    pass
  
  return res

'''
Computes the BFS vision by building the BFS graph and then traversing each layer
returns a dictionary, keys are partitions, values are list of all the depth where the corresponding partition is visited
'''
def BFS_vision(graph,source):
   
  tree = BFS_graph(graph,source)
  vision = {i:[] for i in range(number_of_parts)}
  depth = 0
  for layer in nx.bfs_layers(tree,source):
    if depth == 0:
      pass
    else:
      for node in layer:
        vision[graph.nodes[node]['partition']].append(depth)
    depth += 1
  return vision

'''
Given a BFS vision produces a corresponding dictionary where all depth are weighted and then summed
'''
def BFS_weight(dict):
  return {k: np.sum([1/2**(a-1) for a in v]) if len(v) >= 1 else 0 for k,v in dict.items()}

'''
Given a weighted BFS vision produces the distribution of probabilities of visiting each partition
'''
def BFS_distr(dict):
  total = np.sum(list(dict.values()))
  return {k:v/total if v != 0 else serendipity for k,v in dict.items() }

'''
Gvien a BFS vision produces the distribution of probabilities of visiting each partition
'''
def BFSVISION2DISTR(G,vision):
  return BFS_distr(BFS_weight(vision))

'''
Computes the neighbour vision
returns a dictionary, key are partitions, values are the number of times they appear among all nodes of distance depth
'''
def NEI_vision(graph,source,depth=1):
    vision = {i:serendipity for i in range(number_of_parts)}

    nodelist = nx.bfs_tree(graph,source,depth_limit=depth).nodes()
    for node in nodelist:
        if node != source:
            vision[graph.nodes[node]['partition']] += 1
    
    return vision

'''
Given a neighbour vision produces the distribution of probabilities of visiting each partition
'''
def NEIVISION2DISTR(graph,vision):
    total = np.sum(list(vision.values()))
    return {k:v/total for k,v in vision.items()}

'''
Shortest distance vision of u
for each partition compute the shortest path length from u to it
'''
def SD_vision(graph,source):
    vision = {i:100 for i in range(number_of_parts)}

    for othernode in graph.nodes():
        if othernode != source:
            distance = nx.shortest_path_length(graph,source,othernode)
            if distance < vision[graph.nodes[othernode]['partition']]:
                vision[graph.nodes[othernode]['partition']] = distance
    
    return vision