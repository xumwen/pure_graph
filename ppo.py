import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPS = 1e-5
SCALE = 10
CLIP_EPS = 0.1

nb_episodes = 3
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
        entropy = normal.entropy()

        action = normal.sample()
        log_prob = normal.log_prob(action)

        return state_value, log_prob, entropy


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
            s_value, a_logprob, entropy = self.policy.evaluate(s, a)
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

            loss_1 = -torch.min(surr1, surr2).mean()
            loss_2 = F.mse_loss(s_value, td_target.detach()) * 0.1
            loss_3 = -entropy.mean() * 0.1

            loss = loss_1 + loss_2 + loss_3

            if loss.item() > 100:
                # print('ratio: {:.2f}'.format(ratio))
                # print('advantage: {:.2f}'.format(advantage))
                print('Loss part 1: {:.2f}'.format(loss_1.item()))
                print('Loss part 2: {:.2f}'.format(loss_2.item()))
                print('Loss part 3: {:.2f}'.format(loss_3.item()))
                print('Loss sum: {:.2f}'.format(loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.policy.cpu()

    def train_step(self, env):
        for eid in range(nb_episodes):
            env.reset()
            action_cnt = 0

            # get all subgraphs
            while not env.finish():
                done = False
                s, done = env.get_init_state()
                if done:
                    break

                # get a subgraph step by step
                for step_i in range(env.sample_step):
                    a, logp = self.policy.action(s)
                    s_prime, done = env.step(a, step_i+1)

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

            r = env.eval()
            for i in range(action_cnt):
                self.memory.rewards.append(r)
            self.train()
            self.memory.clear_mem()
