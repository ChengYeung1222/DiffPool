import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler
from graph_sampler import GraphSampler2

def prepare_val_data(graphs, args, val_idx, max_nodes=0):

    random.shuffle(graphs)
    val_size = len(graphs) // 10  #related to train.py 538th row
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch

    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

# modified by niefan
def prepare_val_data2(Hlist,New_G, args, val_idx, max_nodes=0):
    random.seed(4)#todo
    random.shuffle(Hlist)
    val_size = len(Hlist) // 10  #related to train.py 538th row
    train_graphs = Hlist[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + Hlist[(val_idx+1) * val_size :]
    val_graphs = Hlist[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(Hlist))
    print('Number of edges: ', New_G.number_of_edges()*len(Hlist))
    # print('Max, avg, std of graph size: ',
    #         max([G.number_of_nodes() for G in graphs]), ', '
    #         "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
    #         "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch

    dataset_sampler = GraphSampler2(train_graphs, New_G,normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler2(val_graphs, New_G,normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

def prepare_val_data3(Hlist,New_G, args,  max_nodes=0):
    random.seed(4)#todo
    # random.shuffle(Hlist)
    test_size=len(Hlist)
    test_graphs=Hlist[:120]#todo
    print('Num test graphs: ', len(test_graphs))
    print('Number of graphs: ', len(Hlist))
    print('Number of edges: ', New_G.number_of_edges()*len(Hlist))
    # print('Max, avg, std of graph size: ',
    #         max([G.number_of_nodes() for G in graphs]), ', '
    #         "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
    #         "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch

    dataset_sampler = GraphSampler2(test_graphs, New_G,normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return test_dataset_loader ,dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim