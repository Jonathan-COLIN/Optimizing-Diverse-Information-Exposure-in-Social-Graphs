import numpy as np

def simple_entropy(dictionary):
  sum = 0

  for k,v in dictionary.items():
    sum += v * np.log(v)

  return -sum

def cross_entropy(proba,target_distribution = None):

  if target_distribution is None:
    target_distribution = {k:1/len(proba) for k in proba.keys()}

  return np.sum([target_distribution[i]*np.log(target_distribution[i]/proba[i]) for i in proba.keys()])

def sigma_bar(dictionary):
  return np.sum([np.log(p) for _,p in dictionary.items()])


measures = [simple_entropy, cross_entropy, sigma_bar]