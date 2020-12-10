import networkx as nx
import numpy as np
import torch
import torch.utils.data

import util

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        # keep all variables derived with *self* except for *adj_all*
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []

        self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        #if features == 'default':
        self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))  # related to load_date.py 95 rows
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = util.node_dict(G)[u]['feat']
                self.feature_all.append(f)
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>max_deg] = max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(util.node_iter(G)):
                    f[i,:] = util.node_dict(G)[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings,
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in util.node_dict(G)[0]:
                    node_feats = np.array([util.node_dict(G)[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                        np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
            else:
                self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj':adj_padded,
                'feats':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'assign_feats':self.assign_feat_all[idx].copy()}


class GraphSampler2(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''

    def __init__(self, Hlist, New_G,features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        # keep all variables derived with *self* except for *adj_all*
        self.adj = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []

        self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = New_G.number_of_nodes()
        else:
            self.max_num_nodes = max_num_nodes

        # if features == 'default':
        self.feat_dim = util.node_dict(New_G)[0]['feat'].shape[0]
        self.adj = np.array(nx.to_numpy_matrix(New_G))  # related to load_date.py 95 rows

        for i in range(len(Hlist)):
            # if normalize:
            #     sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
            #     adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            #self.adj_all.append(adj)
            self.len_all.append(New_G.number_of_nodes())
            self.label_all.append(Hlist[i][0])
            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                # f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                # for i, u in enumerate(New_G.nodes()):
                #     f[i, :] = util.node_dict(G)[u]['feat']
                self.feature_all.append(Hlist[i][1])

            # elif features == 'id':
            #     self.feature_all.append(np.identity(self.max_num_nodes))
            # elif features == 'deg-num':
            #     degs = np.sum(np.array(adj), 1)
            #     degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - New_G.number_of_nodes()], 0),
            #                           axis=1)
            #     self.feature_all.append(degs)
            # elif features == 'deg':
            #     self.max_deg = 10
            #     degs = np.sum(np.array(adj), 1).astype(int)
            #     degs[degs > max_deg] = max_deg
            #     feat = np.zeros((len(degs), self.max_deg + 1))
            #     feat[np.arange(len(degs)), degs] = 1
            #     feat = np.pad(feat, ((0, self.max_num_nodes - New_G.number_of_nodes()), (0, 0)),
            #                   'constant', constant_values=0)
            #
            #     f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            #     for i, u in enumerate(util.node_iter(G)):
            #         f[i, :] = util.node_dict(G)[u]['feat']
            #
            #     feat = np.concatenate((feat, f), axis=1)
            #
            #     self.feature_all.append(feat)
            # elif features == 'struct':
            #     self.max_deg = 10
            #     degs = np.sum(np.array(adj), 1).astype(int)
            #     degs[degs > 10] = 10
            #     feat = np.zeros((len(degs), self.max_deg + 1))
            #     feat[np.arange(len(degs)), degs] = 1
            #     degs = np.pad(feat, ((0, self.max_num_nodes - New_G.number_of_nodes()), (0, 0)),
            #                   'constant', constant_values=0)
            #
            #     clusterings = np.array(list(nx.clustering(G).values()))
            #     clusterings = np.expand_dims(np.pad(clusterings,
            #                                         [0, self.max_num_nodes - G.number_of_nodes()],
            #                                         'constant'),
            #                                  axis=1)
            #     g_feat = np.hstack([degs, clusterings])
            #     if 'feat' in util.node_dict(G)[0]:
            #         node_feats = np.array([util.node_dict(G)[i]['feat'] for i in range(G.number_of_nodes())])
            #         node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
            #                             'constant')
            #         g_feat = np.hstack([g_feat, node_feats])
            #
            #     self.feature_all.append(g_feat)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                    np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])))
            else:
                self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        #return len(self.adj_all)
        return len(self.label_all)


    def __getitem__(self, idx):
        adj = self.adj
        num_nodes = adj.shape[0]
        # adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        # adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj': adj,
                'feats': self.feature_all[idx].copy(),
                'label': self.label_all[idx],
                'num_nodes': num_nodes,
                'assign_feats': self.assign_feat_all[idx].copy()}