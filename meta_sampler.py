import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import IterableDataset
from torch.distributions import Normal
from torch_geometric.data import Data

import numpy as np
import pandas as pd
import math
import copy


class MetaSampler(object):
    def __init__(self, data, policy, node_emb, subgraph_nodes, sample_step=10, shuffle=True, random_sample=False):
        self.policy = policy
        self.data = data
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index
        self.node_emb = node_emb
        self.subgraph_nodes = subgraph_nodes
        self.sample_step = sample_step
        self.shuffle = shuffle
        self.random_sample = random_sample
        self.subgraphs = []
        self.reset()
    
    def reset(self):
        self.node_visit = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.subgraphs = []
    
    def get_init_nodes(self, num_init_nodes=100):
        left_nodes = np.where(self.node_visit == 0)[0]
        if self.shuffle:
            np.random.shuffle(left_nodes)
        if len(left_nodes) <= num_init_nodes:
            return left_nodes
            
        return left_nodes[:num_init_nodes]

    def get_neighbor(self, n_id):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(row.size(0), dtype=torch.bool)

        # get 1-hop neighbor
        node_mask[n_id] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_n_id = col[edge_mask].numpy()
        
        # remove visited and cent nodes
        tmp_node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        tmp_node_mask[new_n_id] = True
        neighbor_id = np.where(tmp_node_mask & (node_mask==False) & (self.node_visit==False))[0]

        return neighbor_id

    def random_sample_left_nodes(self, n_id, neighbor_id):
        """ 
        Random sample to warm start without node_emb
        and to avoid exceeding subgraph_nodes
        """
        num_left = self.subgraph_nodes - len(n_id)
        np.random.shuffle(neighbor_id)
        if len(neighbor_id) <= num_left:
            sample_n_id = neighbor_id
        else:
            sample_n_id = neighbor_id[:num_left]
        
        return sample_n_id

    def neighbor_sample(self, neighbor_id, action, min_sample_nodes=10):
        """ 
        Neighbor sample calculate kl-divergence between
        node distribution and action distribution
        """
        if len(neighbor_id) <= min_sample_nodes:
            # avoid small sigma1 which leads kl_div to nan
            return neighbor_id
        
        action = self.rescale_action(action)

        # calculate kl-divergense to sample nodes
        mu1 = action
        sigma1 = self.node_emb.std(dim=1).mean()
        mu2 = self.node_emb[neighbor_id].mean(dim=1)
        sigma2 = self.node_emb[neighbor_id].std(dim=1)

        kl_div = (sigma2 / sigma1).log() + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        kl_div = self.rescale_kl_divergence(kl_div)
        weight = torch.exp(-kl_div)

        # print("action:", action)
        # print("kl_div:", kl_div)
        # print("weight:", weight)

        sample_n_id = neighbor_id[torch.bernoulli(weight) == 1]

        # print("sample_num:", len(sample_n_id))

        return sample_n_id
    
    def get_state(self, n_id, neighbor_id):
        cent_emb = self.node_emb[n_id].sum(dim=0)
        neighbor_emb = self.node_emb[neighbor_id].sum(dim=0)

        state = torch.cat([cent_emb, neighbor_emb], dim=0)
        return state.detach()
    
    def rescale_action(self, action):
        # rescale action to get a sample distribution close to nodes_emb distribution
        mu = action

        nodes_mu = self.node_emb.mean(dim=1)
        mu_max = nodes_mu.max()
        mu_min = nodes_mu.min()

        mu = np.tanh(mu) * (mu_max - mu_min) / 2 + (mu_max + mu_min) / 2

        return mu
    
    def rescale_kl_divergence(self, kl_div, scope=2):
        # rescale kl_div to [0, scope]
        # then weight range [exp(-scope), 1]
        kl_div = kl_div - kl_div.min()
        kl_div = kl_div / kl_div.max() * scope

        return kl_div

    def __produce_subgraph_by_nodes__(self, n_id):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(row.size(0), dtype=torch.bool)

        # visited nodes update
        self.node_visit[n_id] = True

        # get edge
        node_mask[n_id] = True
        edge_mask = node_mask[row] & node_mask[col]
        e_id = np.where(edge_mask)[0]

        # edge_index reindex by n_id
        tmp = torch.empty(self.num_nodes, dtype=torch.long)
        tmp[n_id] = torch.arange(len(n_id))
        edge_index = tmp[self.edge_index[:, e_id]]

        # return data
        data = copy.copy(self.data)
        data.edge_index = edge_index

        N, E = data.num_nodes, data.num_edges
        for key, item in data:
            if item.size(0) == N:
                data[key] = item[n_id]
            if item.size(0) == E:
                data[key] = item[e_id]
        data.n_id = n_id
        data.e_id = e_id

        # print("subgraph:", len(n_id))

        return data

    def __produce_subgraph__(self):
        if self.num_nodes - self.node_visit.sum() <= self.subgraph_nodes:
            # last subgraph
            n_id = np.where(self.node_visit == False)[0]
            return self.__produce_subgraph_by_nodes__(n_id)

        # sample several steps
        n_id = self.get_init_nodes()

        for i in range(self.sample_step):
            neighbor_id = self.get_neighbor(n_id)
            if self.random_sample:
                sample_n_id = self.random_sample_left_nodes(n_id, neighbor_id)
            else:
                s = self.get_state(n_id, neighbor_id)
                action, logp = self.policy.action(s)
                sample_n_id = self.neighbor_sample(neighbor_id, action)
                sample_n_id = self.random_sample_left_nodes(n_id, sample_n_id)
            
            n_id = np.union1d(n_id, sample_n_id)
            # print("n_id:", n_id.shape)
            # print("neighbor_id:", neighbor_id.shape)
            # print("sample_id:", sample_n_id.shape)
            if len(n_id) >= self.subgraph_nodes or len(sample_n_id) == 0:
                break

        return self.__produce_subgraph_by_nodes__(n_id)

    def sample_subgraphs(self):
        self.reset()
        while self.node_visit.sum() != self.num_nodes:
            self.subgraphs.append(self.__produce_subgraph__())