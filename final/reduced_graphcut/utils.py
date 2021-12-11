#!/usr/bin/env python
# coding: utf-8

# reference: modified from https://github.com/NathanZabriskie/GraphCut

# In[4]:


import cv2
import numpy as np
import sys
import maxflow


# In[5]:


def find_averages(b_seeds, f_seeds,img):
    graph = np.zeros((img.shape[0], img.shape[1]))
    graph.fill(0.5)
    for coordinate in b_seeds:
        graph[coordinate[1] - 1, coordinate[0] - 1] = 0

    for coordinate in f_seeds:
        graph[coordinate[1] - 1, coordinate[0] - 1] = 1
    return graph
def get_node_num(x, y, array_shape):
    return y * array_shape[1] + x
def grad_to_weight(gradient, mode=1):
    if mode == 0:
        return np.exp(-(gradient ** 2) / 16)

    if mode == 1:
        return 1 / (1 + gradient)
def get_gradient(pos1, pos2, img):
    x1, y1 = pos1
    x2, y2 = pos2
    return np.sum(np.square(img[y1, x1] - img[y2, x2]))
def populate_graph(graph,img):
    nodes = []
    edges = []

    # make all s and t connections for the graph
    for (y, x), value in np.ndenumerate(graph):
        # this is a background pixel
        if value == 0.0:
            nodes.append((get_node_num(x, y, img.shape), sys.maxsize, 0))
        # this is a foreground node
        elif value == 1.0:
            nodes.append((get_node_num(x, y, img.shape), 0, sys.maxsize))
        else:
            nodes.append((get_node_num(x, y, img.shape), 0, 0))
    for (y, x), value in np.ndenumerate(graph):
        if y == graph.shape[0] - 1 or x == graph.shape[1] - 1:
            continue
        my_index = get_node_num(x, y, img.shape)
        neighbor_index = get_node_num(x + 1, y,img.shape)
        g = grad_to_weight(get_gradient((x, y), (x + 1, y), img))
        edges.append((my_index, neighbor_index, g))

        neighbor_index = get_node_num(x, y + 1, img.shape)
        g = grad_to_weight(get_gradient((x, y), (x, y + 1), img))
        edges.append((my_index, neighbor_index, g))
    return edges, nodes


# In[6]:


def get_xy(nodenum, array_shape):
    return int(nodenum % array_shape[1]), int(nodenum / array_shape[1])
def cut_graph(nodes,edges,img):
    segment_overlay = np.zeros_like(img)
    mask = np.zeros_like(img, dtype=bool)
    g = maxflow.Graph[float](len(nodes), len(edges))
    nodelist = g.add_nodes(len(nodes))

    for node in nodes:
        g.add_tedge(nodelist[node[0]], node[1], node[2])

    for edge in edges:
        g.add_edge(edge[0], edge[1], edge[2], edge[2])

    flow = g.maxflow()

    for index in range(len(nodes)):
        if g.get_segment(index) == 1:
            xy = get_xy(index, img.shape)
            segment_overlay[xy[1], xy[0]] = (255, 0, 255)
            mask[xy[1], xy[0]] = (True, True, True)
    return mask, segment_overlay


# In[ ]:





# In[ ]:




