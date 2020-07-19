import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader

from meta_sampler import MetaSampler

class MetaSampleEnv():
    def __init__(self, data, model, node_emb, node_df, label_matrix, args, device):
        self.model = model
        self.data = data
        self.num_nodes = data.num_nodes

        # task required
        self.is_multi = args.is_multi
        self.node_df = node_df
        self.label_matrix = label_matrix
        self.train_nid = self.data.indices[data.train_mask].numpy()
        self.test_nid = self.data.indices[data.test_mask].numpy()
        self.val_nid = self.data.indices[data.val_mask].numpy()

        # meta sampler required
        self.node_emb = node_emb
        self.subgraph_nodes = args.subgraph_nodes
        self.sample_step = args.sample_step

        self.device = device
    
    def reset(self):
        self.meta_sampler = MetaSampler(self.data, policy=None, node_emb=self.node_emb,
                                        subgraph_nodes=self.subgraph_nodes, 
                                        sample_step=self.sample_step)
    
    def get_state(self):
        """get state vector of current subgraph"""
        cent_emb = self.node_emb[self.n_id].sum(dim=0)
        neighbor_emb = self.node_emb[self.neighbor_id].sum(dim=0)

        state = torch.cat([cent_emb, neighbor_emb], dim=0)
        return state.detach().cpu()
    
    def get_init_state(self):
        """Random init nodes and get neighbor, return state"""
        done = False
        unvisited_nodes_num = self.num_nodes - self.meta_sampler.node_visit.sum()
        if unvisited_nodes_num <= self.subgraph_nodes:
            # last subgraph
            self.n_id = np.where(self.meta_sampler.node_visit == False)[0]
            self.neighbor_id = self.n_id
            done = True
            
            subgraph = self.meta_sampler.__produce_subgraph_by_nodes__(self.n_id)
            self.meta_sampler.subgraphs.append(subgraph)
        else:
            self.n_id = self.meta_sampler.get_init_nodes()
            self.neighbor_id = self.meta_sampler.get_neighbor(self.n_id)
        
        s = self.get_state()

        return s, done
    
    def step(self, action, num_step):
        """extend current subgraph with an action"""
        sample_n_id = self.meta_sampler.neighbor_sample(self.neighbor_id, action)
        # avoid exceeding subgraph_nodes(always intend to sample all neighbors)
        sample_n_id = self.meta_sampler.random_sample_left_nodes(self.n_id, sample_n_id)

        self.n_id = np.union1d(self.n_id, sample_n_id)
        done = False

        if len(self.n_id) >= self.subgraph_nodes or num_step == self.sample_step:
            # last extension
            self.neighbor_id = self.n_id
            done = True

            subgraph = self.meta_sampler.__produce_subgraph_by_nodes__(self.n_id)
            self.meta_sampler.subgraphs.append(subgraph)
        else:
            self.neighbor_id = self.meta_sampler.get_neighbor(self.n_id)
        
        s_prime = self.get_state()

        return s_prime, done
    
    def finish(self):
        """subgraph nodes cover the whole graph"""
        if self.meta_sampler.node_visit.sum() == self.num_nodes:
            return True
        return False
    
    def eval(self):
        """evaluation and return validation f1-micro as reward"""
        subgraph = self.meta_sampler.subgraphs[-1]
        if self.is_multi:
            r = self.eval_sample_multi(subgraph)
        else:
            r = self.eval_sample(subgraph)
        
        return r

    def eval_sample(self, subgraph):
        self.model.zero_grad()
        self.model.eval()

        subgraph = subgraph.to(self.device)
        out, _ = self.model(subgraph.x, subgraph.edge_index)

        out = out.log_softmax(dim=-1)
        pred = out.argmax(dim=-1).cpu().numpy()

        val_mask = np.zeros(self.num_nodes, dtype=bool)
        val_mask[self.val_nid] = True
        indices = subgraph.indices.cpu().numpy()
        mask = val_mask[indices]

        val_acc = accuracy_score(self.node_df['y'][indices][mask], pred[mask])

        self.model.train()

        return val_acc
    
    def eval_sample_multi(self, subgraph):
        self.model.zero_grad()
        self.model.eval()

        subgraph = subgraph.to(self.device)
        out, _ = self.model(subgraph.x, subgraph.edge_index)
        pred = (out > 0).float().cpu().numpy()

        val_mask = np.zeros(self.num_nodes, dtype=bool)
        val_mask[self.val_nid] = True
        indices = subgraph.indices.cpu().numpy()
        mask = val_mask[indices]
        val_acc = f1_score(self.label_matrix[indices][mask], pred[mask], average='micro')
        
        self.model.train()

        return val_acc