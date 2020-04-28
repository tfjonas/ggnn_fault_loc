import torch
import numpy as np

def biggest_bet_error(Y,T,D,list_all=False):
    error = []
    predictions = torch.exp(Y).argmax(-1)
    for elem in range(len(T)):
        error.append((D[elem,T[elem],predictions[elem]]/D[elem].max()).item())
    if list_all==True:
        return error    
    else:
        return min(error), sum(error)/len(error), max(error)

def weighted_bets_error(Y,T,D,list_all=False):
    error = []
    for elem in range(len(T)):        
        e = torch.sum((D[elem,T[elem]])*torch.exp(Y[elem])/D[elem].max(),dim=-1)
        error.append(e.item())
    if list_all==True:
        return error    
    else:
        return min(error), sum(error)/len(error), max(error)
