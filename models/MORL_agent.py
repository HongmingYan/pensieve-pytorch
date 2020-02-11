# This is from Runzhe_MORL/multimario/agent.py
# need to change this into MORL_A3C.py
# import NN from MORL_NN

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

from MORL_NN import *

import torch.optim as optim


class NaiveMoActorAgent(object):
    def __init__(
            self,
            args,
            input_size,
            output_size,
            reward_size):
        self.model = NaiveMoCnnActorCriticNetwork(
            input_size, output_size, reward_size)
        self.num_env = args.num_worker
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = args.num_step
        self.use_gae = args.use_gae
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.learning_rate)
        self.standardization = args.standardization
        self.entropy_coef = args.entropy_coef
        self.clip_grad_norm = args.clip_grad_norm

        self.device = torch.device('cuda' if args.use_cuda else 'cpu')

        self.model = self.model.to(self.device)

    def get_action(self, state, preference):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        w = torch.Tensor(preference).to(self.device)
        w = w.float()
        policy, value = self.model(state, w)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(policy)

        return action

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def forward_transition(self, state, next_state, preference):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        w = torch.Tensor(preference).to(self.device)
        w = w.float()
        policy, value = self.model(state, w)

        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        w = torch.Tensor(preference).to(self.device)
        w = w.float()
        _, next_value = self.model(next_state, w)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy

    def find_preference(
            self,
            w_batch,
            target_batch,
            pref_param):
        with torch.no_grad():
            w_batch = torch.FloatTensor(w_batch).to(self.device)
            target_batch = torch.FloatTensor(pref_param).to(self.device) / 200

        # compute loss
        pref_param = torch.FloatTensor(pref_param).to(self.device)
        pref_param.requires_grad = True
        sigmas = torch.FloatTensor([0.01] * len(pref_param)).to(self.device)
        dist = torch.distributions.normal.Normal(pref_param, sigmas)
        pref_loss = dist.log_prob(w_batch).sum(dim=1) * target_batch

        self.optimizer.zero_grad()
        # Total loss
        loss = pref_loss.mean()
        loss.backward()

        eta = 2.0
        pref_param = pref_param + eta * pref_param.grad
        pref_param = simplex_proj(pref_param.detach().cpu().cpu().numpy())
        print("update prefreence parameters to", pref_param)

        return pref_param

    def train_model(
            self,
            s_batch,
            next_s_batch,
            w_batch,
            target_batch,
            action_batch,
            adv_batch):
        with torch.no_grad():
            s_batch = torch.FloatTensor(s_batch).to(self.device)
            next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
            w_batch = torch.FloatTensor(w_batch).to(self.device)
            target_batch = torch.FloatTensor(target_batch).to(self.device)
            action_batch = torch.LongTensor(action_batch).to(self.device)
            adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        if self.standardization:
            adv_batch = (adv_batch - adv_batch.mean()) / \
                        (adv_batch.std() + 1e-30)

        ce = nn.CrossEntropyLoss()
        # mse = nn.SmoothL1Loss()
        forward_mse = nn.MSELoss()

        # for multiply advantage
        policy, value = self.model(s_batch, w_batch)
        m = Categorical(F.softmax(policy, dim=-1))

        # Actor loss
        actor_loss = -m.log_prob(action_batch) * adv_batch

        # Entropy(for more exploration)
        entropy = m.entropy()

        # Critic loss
        mse = nn.MSELoss()
        critic_loss = mse(value.sum(1), target_batch)

        self.optimizer.zero_grad()

        # Total loss
        loss = actor_loss.mean() + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

# projection to simplex
def simplex_proj(x):
    y = -np.sort(-x)
    sum = 0
    ind = []
    for j in range(len(x)):
        sum = sum + y[j]
        if y[j] + (1 - sum) / (j + 1) > 0:
            ind.append(j)
        else:
            ind.append(0)
    rho = np.argmax(ind)
    delta = (1 - (y[:rho+1]).sum())/(rho+1)
    return np.clip(x + delta, 0, 1)