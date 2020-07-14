import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from meta_sampler import MetaSampler


NUM_INIT_NODES = 10

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPS = 1e-5
SCALE = 10
CLIP_EPS = 0.1

nb_episodes = 3
nb_steps = 10
nb_epoches = 3

gamma = 0.98
lambda_ = 0.95


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.logprobs = []
        self.done_lst = []

    def clear_mem(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.done_lst[:]


class MetaSampleEnv():
    def __init__(self, data, model, node_emb, node_df, args, device):
        self.model = model
        self.data = data
        self.num_nodes = data.num_nodes
        self.node_emb = node_emb
        self.node_df = node_df
        self.subgraph_nodes = args.subgraph_nodes
        self.sample_step = args.sample_step
        self.norm_loss = args.loss_norm
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
    
    def step(self, action):
        """extend current subgraph with an action"""
        sample_n_id = self.meta_sampler.neighbor_sample(self.neighbor_id, action)
        # avoid over subgraph_nodes(always intend to sample all neighbors)
        sample_n_id = self.meta_sampler.random_sample_left_nodes(self.n_id, sample_n_id)

        self.n_id = np.union1d(self.n_id, sample_n_id)
        done = False

        if len(self.n_id) >= self.subgraph_nodes:
            # last subgraph
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

    def eval(self, episode):
        """evaluation and return loss as reward"""
        self.model.zero_grad()
        self.model.eval()

        loader = self.meta_sampler.subgraphs

        res_df_list = []
        for data in loader:
            data.to(self.device)

            if self.norm_loss == 1:
                out, _ = self.model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
            else:
                out, _ = self.model(data.x, data.edge_index)
            out = out.log_softmax(dim=-1)
            pred = out.argmax(dim=-1)

            res_batch = pd.DataFrame()
            res_batch['nid'] = data.indices.cpu().numpy()
            res_batch['pred'] = pred.cpu().numpy()
            res_df_list.append(res_batch)

            res_df_duplicate = pd.concat(res_df_list)
            tmp = res_df_duplicate.groupby(['nid', 'pred']).size().unstack().fillna(0)
            res_df = pd.DataFrame()
            res_df['nid'] = tmp.index
            res_df['pred'] = tmp.values.argmax(axis=1)

            res_df.columns = ['nid', 'pred']
            res_df = res_df.merge(self.node_df, on=['nid'], how='left')

            accs = res_df.groupby(['mask']).apply(lambda x: accuracy_score(x['y'], x['pred'])).reset_index()
            accs.columns = ['mask', 'acc']
            accs = accs.sort_values(by=['mask'], ascending=True)
            accs = accs['acc'].values
        
        self.model.train()

        return accs[1]
    

class ActorCritic(nn.Module):
    def __init__(self, state_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.build_actor()
        self.build_critic()

    def build_actor(self):
        self.actor = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU()
        )
        self.action_mean = nn.Linear(32, 1)
        self.action_log_std = nn.Linear(32, 1)
    
    def action(self, state):
        act_hid = self.actor(state)

        action_mean = self.action_mean(act_hid)
        action_log_std = self.action_log_std(act_hid)

        action_log_std = torch.clamp(action_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        action_std = action_log_std.exp()

        normal = Normal(action_mean, action_std)

        action = normal.sample()
        logprob = normal.log_prob(action)

        return action.item(), logprob.item()

    def build_critic(self):
        # v network
        self.critic = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def evaluate(self, state, action):
        # return the value of given state, and the probability of the actor take {action}
        state_value = self.critic(state)

        if action is None:
            return state_value, None

        act_hid = self.actor(state)

        action_mean = self.action_mean(act_hid)
        action_log_std = self.action_log_std(act_hid)

        action_log_std = torch.clamp(action_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        action_std = action_log_std.exp()

        normal = Normal(action_mean, action_std)

        action = normal.sample()
        action_logprob = normal.log_prob(action)

        return state_value, action_logprob


class PPO:
    def __init__(self, state_size, device):
        self.policy = ActorCritic(state_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.memory = Memory()
        self.device = device

    def make_batch(self):
        # Data format
        # state is a list of tensor
        # others are a list of single value
        s = torch.stack(self.memory.states).to(self.device)
        a = torch.FloatTensor(self.memory.actions).to(self.device).unsqueeze(dim=1)
        r = torch.FloatTensor(self.memory.rewards).to(self.device).unsqueeze(dim=1)
        s_prime = torch.stack(self.memory.next_states).to(self.device)
        logp = torch.FloatTensor(self.memory.logprobs).to(self.device).unsqueeze(dim=1)
        done_mask = torch.FloatTensor(self.memory.done_lst).to(self.device).unsqueeze(dim=1)

        return s, a, r, s_prime, logp, done_mask

    def train(self):
        s, a, r, s_prime, logp, done_mask = self.make_batch()
        self.policy.to(self.device)

        for i in range(nb_epoches):
            s_value, a_logprob = self.policy.evaluate(s, a)
            s_prime_value, _ = self.policy.evaluate(s_prime, None)

            td_target = r + gamma * s_prime_value * done_mask
            delta = td_target - s_value
            delta = delta.detach().cpu().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lambda_ * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.FloatTensor(advantage_list).to(self.device)

            ratio = torch.exp(a_logprob - logp)

            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1 - CLIP_EPS,
                1 + CLIP_EPS
            ) * advantage

            loss = -torch.min(surr1, surr2) + \
                F.l1_loss(s_value, td_target.detach())

            if loss.mean().item() > 100:
                # print('ratio: ', ratio)
                # print('advantage: ', advantage)
                print('Loss part 1: ', -torch.min(surr1, surr2).mean().item())
                print('Loss part 2: ', F.l1_loss(s_value, td_target.detach()).mean().item())
                print('Loss sum: ', loss.mean().item())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy.cpu()

    def train_step(self, env):
        # print("ppo train start")
        for eid in range(nb_episodes):
            # print("episode:", eid+1)
            env.reset()
            action_cnt = 0

            # get all subgraphs
            while not env.finish():
                done = False
                s, done = env.get_init_state()
                if done:
                    break

                # get a subgraph step by step
                for i in range(nb_steps):
                    a, logp = self.policy.action(s)
                    s_prime, done = env.step(a)

                    self.memory.states.append(s)
                    self.memory.actions.append(a)
                    self.memory.next_states.append(s_prime)
                    self.memory.logprobs.append(logp)
                    done_mask = 0 if done else 1
                    self.memory.done_lst.append(done_mask)

                    s = s_prime
                    action_cnt += 1

                    if done:
                        break

            r = env.eval(eid+1)
            for i in range(action_cnt):
                self.memory.rewards.append(r)
            self.train()
            self.memory.clear_mem()
        # print("ppo train end")
