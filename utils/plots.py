## -*- coding: utf-8 -*-
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_graph_edges(G, devices):
    nx.set_edge_attributes(G,'black','color')
    nx.set_edge_attributes(G,0.5,'width')
    next_device = []
    for edge in nx.dfs_edges(G, list(devices).index(0.5)):
        if devices[edge[0]] == 1.5:
            next_device.append(edge[0])
        G.edges[edge]['color'] = 'blue'
        G.edges[edge]['width'] = 1
    for dev in next_device:
        for edge in nx.dfs_edges(G, dev):
            G.edges[edge]['color'] = 'black'
            G.edges[edge]['width'] = 0.5    
        
    pos = nx.get_node_attributes(G,'pos')
    colors = nx.get_edge_attributes(G,'color')
    nx.draw_networkx_edges(G, pos, arrows=False,alpha=1.0,
                        width = [G.edges[edge]['width'] for edge in G.edges],
                        edge_color = [G.edges[edge]['color'] for edge in G.edges])
                        
def plot_pred(G, pred, target, devices):
    plt.axes().set_aspect(1.0)  
    
    G = nx.OrderedDiGraph(G, nodelist=range(len(G)))
    plot_graph_edges(G, devices)

    pos=nx.get_node_attributes(G,'pos')
  
    # Fault
    faultpos = G.nodes[target]['pos']
    plt.scatter(faultpos[0], faultpos[1], marker='x', color='red',
                s=130, label='Falta', zorder=25)
           
    # Devices    
    colors = list(map({0.5: 'red', 1: 'None', 1.5: 'black'}.get, devices[:len(G)]))
    nx.set_node_attributes(G, dict(zip(range(len(G)), colors)), 'color')
    nx.set_node_attributes(G, 50, 'size')
    G.nodes[0]['size'] = 150
    ax = nx.draw_networkx_nodes(G, pos, node_shape='s',
                                node_size=[G.nodes[node]['size'] for node in G], 
                                node_color=[G.nodes[node]['color'] for node in G], zorder=20)

    pred = np.array([pred[n] for n in list(G)])
    cmap=plt.cm.viridis
    vmin = 0
    vmax = 1

    ax = nx.draw_networkx_nodes(G, pos, node_size=5000*pred, 
                                alpha=0.2, cmap=cmap, node_color=pred,
                                vmin=vmin, vmax=vmax, zorder=500)
    
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #sm._A = []
    #plt.colorbar(sm,fraction=0.022, pad=0.02, orientation='horizontal')    
    plt.axis('off') 
    
def plot_isc(G):
    plt.axes().set_aspect(1.0)
    G = nx.OrderedDiGraph(G, nodelist=range(len(G)))
    pos = nx.get_node_attributes(G,'pos')

    # edges
    edg = nx.draw(G, pos, node_size=0, arrows=False)

    # nodes
    isc1 = np.array([G.nodes[node]['isc1'] for node in G])
    isc1 = (isc1-min(isc1))/(max(isc1-min(isc1)))

    cmap=plt.cm.viridis
    vmin, vmax = 0, 1
    nds = nx.draw_networkx_nodes(G, pos, node_size=100*isc1, cmap=cmap, node_color=isc1,
                                 vmin=vmin, vmax=vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm,fraction=0.022, pad=0.02, orientation='horizontal')
    plt.axis('off')                        
