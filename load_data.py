import networkx as nx
import numpy as np
import scipy as sc
import os
import re

import util


def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}                   # {} 表示字典
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")    #strip("\n") 去掉换行符号
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1] #俩个列表之间直接相加
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))

    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset 
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    #underlying is the source code

    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}  #
    # common_adj = [] #modify by zy
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}  #
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))  #

            #if e1 > max_nodes:           #modify by zy
                #break                    #modify by zy
            #common_adj.append((e0, e1))  #modify by zy
            index_graph[graph_indic[e0]]+=[e0,e1]#
            num_edges += 1
    for k in index_graph.keys():
         index_graph[k]=[u-1 for u in set(index_graph[k])]


    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i]) # related to graph_sampler.py 29 rows
        #print(isinstance(G,Dictionary))
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        for u in util.node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                util.node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                util.node_dict(G)[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}



        it = 0
        for n in util.node_iter(G):
            mapping[n] = it
            it += 1
        #NewG=nx.relabel_nodes(G,mapping)
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs

    # common_adj = []
    # num_edges = 0
    # with open(filename_adj) as f:
    #     for line in f:
    #         line = line.strip("\n").split(",")
    #         e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
    #         if e1 > max_nodes:
    #             break
    #         common_adj.append((e0, e1))
    #         num_edges += 1
    #
    # Hlist=[]
    # for i in range(1, 1 + len(graph_labels)):
    #     # indexed from 1 here
    #     LAlist = []
    #     #LAlist[]用于存放标签和属性特征
    #     G = nx.from_edgelist(common_adj)  # related to graph_sampler.py 29 rows
    #     # print(type(G))
    #     if max_nodes is not None and G.number_of_nodes() > max_nodes:
    #         continue
    #
    #     # add features and labels
    #     G.graph['label'] = graph_labels[i - 1]
    #     LAlist.append(G.graph['label'])
    #
    #     TList = []
    #     for u in util.node_iter(G):
    #
    #         # if len(node_labels) > 0:
    #         #     node_label_one_hot = [0] * num_unique_node_labels
    #         #     node_label = node_labels[u - 1]
    #         #     node_label_one_hot[node_label] = 1
    #         #     util.node_dict(G)[u]['label'] = node_label_one_hot
    #         if len(node_attrs) > 0:
    #             util.node_dict(G)[u]['feat'] = node_attrs[(u - 1)+(i-1)*2592]
    #             TList.append(util.node_dict(G)[u]['feat'])
    #     LAlist.append(np.array(TList))
    #
    #
    #     if len(node_attrs) > 0:
    #         G.graph['feat_dim'] = node_attrs[0].shape[0]
    #
    #     # relabeling
    #     mapping = {}
    #     it = 0
    #     for n in util.node_iter(G):
    #         mapping[n] = it
    #         it += 1
    #     New_G=nx.relabel_nodes(G, mapping)
    #     # indexed from 0
    #     Hlist.append(LAlist)
    # return Hlist,New_G

def read_graphfile2(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}                   # {} 表示字典
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")    #strip("\n") 去掉换行符号
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1] #俩个列表之间直接相加
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))

    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    #underlying is the source code

    # adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}  #
    # # common_adj = [] #modify by zy
    # index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}  #
    # num_edges = 0
    # with open(filename_adj) as f:
    #     for line in f:
    #         line = line.strip("\n").split(",")
    #         e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
    #         adj_list[graph_indic[e0]].append((e0, e1))  #
    #
    #         #if e1 > max_nodes:           #modify by zy
    #             #break                    #modify by zy
    #         #common_adj.append((e0, e1))  #modify by zy
    #         index_graph[graph_indic[e0]]+=[e0,e1]#
    #         num_edges += 1
    # for k in index_graph.keys():
    #      index_graph[k]=[u-1 for u in set(index_graph[k])]
    #
    #
    # graphs = []
    # for i in range(1, 1 + len(adj_list)):
    #     # indexed from 1 here
    #     G=nx.from_edgelist(adj_list[i]) # related to graph_sampler.py 29 rows
    #     #print(isinstance(G,Dictionary))
    #     if max_nodes is not None and G.number_of_nodes() > max_nodes:
    #         continue
    #
    #     # add features and labels
    #     G.graph['label'] = graph_labels[i - 1]
    #     for u in util.node_iter(G):
    #         if len(node_labels) > 0:
    #             node_label_one_hot = [0] * num_unique_node_labels
    #             node_label = node_labels[u - 1]
    #             node_label_one_hot[node_label] = 1
    #             util.node_dict(G)[u]['label'] = node_label_one_hot
    #         if len(node_attrs) > 0:
    #             util.node_dict(G)[u]['feat'] = node_attrs[u - 1]
    #     if len(node_attrs) > 0:
    #         G.graph['feat_dim'] = node_attrs[0].shape[0]
    #
    #     # relabeling
    #     mapping = {}
    #
    #
    #
    #     it = 0
    #     for n in util.node_iter(G):
    #         mapping[n] = it
    #         it += 1
    #     #NewG=nx.relabel_nodes(G,mapping)
    #     # indexed from 0
    #     graphs.append(nx.relabel_nodes(G, mapping))
    # return graphs

    common_adj = []
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            if e1 > max_nodes:
                break
            common_adj.append((e0, e1))
            num_edges += 1

    Hlist=[]
    for i in range(1, 1 + len(graph_labels)):
        # indexed from 1 here
        LAlist = []
        #LAlist[]用于存放标签和属性特征
        G = nx.from_edgelist(common_adj)  # related to graph_sampler.py 29 rows
        # print(type(G))
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        LAlist.append(G.graph['label'])

        TList = []
        for u in util.node_iter(G):

            # if len(node_labels) > 0:
            #     node_label_one_hot = [0] * num_unique_node_labels
            #     node_label = node_labels[u - 1]
            #     node_label_one_hot[node_label] = 1
            #     util.node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                util.node_dict(G)[u]['feat'] = node_attrs[(u - 1)+(i-1)*2592]
                TList.append(util.node_dict(G)[u]['feat'])
        LAlist.append(np.array(TList))


        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        for n in util.node_iter(G):
            mapping[n] = it
            it += 1
        New_G=nx.relabel_nodes(G, mapping)
        # indexed from 0
        Hlist.append(LAlist)
    return Hlist,New_G