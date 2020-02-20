import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score

from scipy import stats

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPS = 1e-5
SCALE = 1
CLIP_EPS = 0.1

nb_bins = 10
state_dim = nb_bins * 2

nb_episodes = 100
nb_sample_rounds = 10 - 1
nb_epoches = 5

gamma = 0.98
lambda_ = 0.95

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def norm_pdf(x, mu, sigma):
    s1 = 1.0 / sigma / np.sqrt(2.0 * np.pi)
    s2 = -((x - mu) ** 2) / 2 / (sigma ** 2)
    return s1 * np.exp(s2)


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.logprobs = []

    def clear_mem(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.logprobs[:]


class ResampleEnv:
    def __init__(self, nb_bins=nb_bins):
        self._init_dataset()
        self._init_cuts()

        self.nb_bins = nb_bins
        self.models = []

    def _init_cuts(self):
        self.cuts = np.linspace(0.0, 1.0, nb_bins + 1)
        self.cuts[-1] += 1e-8

    def _init_dataset(self):
        train_data = pd.read_csv('./data/credit_train.csv').\
            drop('Unnamed: 0', axis=1)
        valid_data = pd.read_csv('./data/credit_valid.csv').\
            drop('Unnamed: 0', axis=1)
        test_data = pd.read_csv('./data/credit_test.csv').\
            drop('Unnamed: 0', axis=1)

        self.train_X, self.train_y = train_data.iloc[:, :-1], \
            train_data.iloc[:, -1]
        self.valid_X, self.valid_y = valid_data.iloc[:, :-1], \
            valid_data.iloc[:, -1]
        self.test_X, self.test_y = test_data.iloc[:, :-1], \
            test_data.iloc[:, -1]

    def _get_bin(self, x):
        for i in range(len(self.cuts) - 1):
            if self.cuts[i] <= x < self.cuts[i + 1]:
                return i
        assert(True)

    def eval_loss(self, X, y):
        loss_df = pd.DataFrame()
        loss_df['row_id'] = np.arange(X.shape[0])
        loss_df['label'] = y.values
        loss_df['prob'] = 0

        for model in self.models:
            loss_df['prob'] += model.predict_proba(X)[:, 1]
        if len(self.models) > 1:
            loss_df['prob'] /= len(self.models)

        loss_df['loss'] = (loss_df['label'] - loss_df['prob']).abs()

        loss_df['bin'] = loss_df['loss'].map(lambda x: self._get_bin(x))

        return loss_df

    def _population_dist(self, loss_df):
        pop_dict = loss_df.groupby('bin').size().to_dict()
        pop_arr = np.asarray([pop_dict.get(i, 0.0)
                              for i in range(self.nb_bins)]).astype('float')

        pop_arr /= pop_arr.sum()
        return pop_arr

    def _error_dist(self, loss_df):
        err_dict = loss_df.groupby('bin').apply(
            lambda x: x['loss'].sum()).to_dict()
        err_arr = np.asarray([err_dict.get(i, 0.0)
                              for i in range(self.nb_bins)]).astype('float')
        err_arr /= err_arr.sum()
        return err_arr

    def _fit_model(self, X, y, index_pos, index_neg):
        train_X = pd.concat(
            [X.loc[index_pos], X.loc[index_neg]], axis=0
        )
        train_y = pd.concat(
            [y.loc[index_pos], y.loc[index_neg]], axis=0
        )
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(train_X, train_y)
        return model

    def _rus_index(self, X, y):
        index_pos = X[y == 1].index
        index_neg = X[y == 0].sample(len(index_pos) * SCALE).index
        return index_pos, index_neg

    # random under sample
    def _rus(self, X, y):
        index_pos, index_neg = self._rus_index(X, y)
        return self._fit_model(X, y, index_pos, index_neg)

    def _gus_index(self, X, y, mu, sigma):
        loss_df = self.eval_loss(X, y)
        loss_df['weight'] = loss_df['loss'].map(
            lambda x: norm_pdf(x, mu, sigma)
        )

        index_pos = X[y == 1].index

        loss_df = loss_df.loc[X[y == 0].index]
        index_neg = loss_df.sample(
            len(index_pos) * SCALE, weights=loss_df['weight']).index

        return index_pos, index_neg

    # gaussian under sample
    def _gus(self, X, y, mu, sigma=0.2):
        index_pos, index_neg = self._gus_index(X, y, mu, sigma)
        return self._fit_model(X, y, index_pos, index_neg)

    def _get_state(self):
        train_loss_df = self.eval_loss(self.train_X, self.train_y)
        valid_loss_df = self.eval_loss(self.valid_X, self.valid_y)

        train_pop_dist = self._population_dist(train_loss_df)
        train_err_dist = self._error_dist(train_loss_df)

        valid_pop_dist = self._population_dist(valid_loss_df)
        valid_err_dist = self._error_dist(valid_loss_df)

        return np.concatenate(
            [train_pop_dist, train_err_dist, valid_pop_dist, valid_err_dist], axis=0
        )

    def _score(self, X, y):
        loss_df = self.eval_loss(X, y)
        return average_precision_score(loss_df['label'], loss_df['prob'])

    def score(self, on_eval='test'):
        if on_eval == 'train':
            return self._score(self.train_X, self.train_y)
        elif on_eval == 'valid':
            return self._score(self.valid_X, self.valid_y)
        elif on_eval == 'test':
            return self._score(self.test_X, self.test_y)

    def reset(self):
        del self.models[:]

        self.models.append(
            self._rus(self.train_X, self.train_y)
        )

        state = self._get_state()

        return state

    def _rescale_action(self, action):
        action = np.tanh(action)
        return action * 0.5 + 0.5

    def step(self, mu):
        mu = self._rescale_action(mu)
        old_score = self.score(on_eval='valid')

        self.models.append(
            self._gus(self.train_X, self.train_y, mu)
        )

        state = self._get_state()
        valid_score = self.score(on_eval='valid')
        test_score = self.score(on_eval='test')

        print('Action: {}, Valid Score {}, Test Score {}'.format(
            mu, valid_score, test_score
        ))

        return state, valid_score - old_score


class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim

        self._build_actor()
        self._build_critic()

    def _build_actor(self):
        # pi network
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )
        self.actor_mean = nn.Linear(32, 1)
        self.actor_log_std = nn.Linear(32, 1)

    def _build_critic(self):
        # v network
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def act(self, state):
        act_hid = self.actor(state)

        mean = self.actor_mean(act_hid)
        log_std = self.actor_log_std(act_hid)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)

        action = torch.tanh(normal.sample())
        logprob = normal.log_prob(action)

        return action.item(), logprob.item()

    def evaluate(self, state, action):
        state_value = self.critic(state)

        if action is None:
            return state_value, None

        act_hid = self.actor(state)

        mean = self.actor_mean(act_hid)
        log_std = self.actor_log_std(act_hid)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)

        action_logprob = normal.log_prob(action)

        return state_value, action_logprob


class PPO:
    def __init__(self, state_dim):
        self.policy = ActorCritic(state_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.memory = Memory()

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        # make batch dim
        state = state.unsqueeze(dim=0)
        return self.policy.act(state)

    def make_batch(self):
        s = torch.FloatTensor(self.memory.states).to(device)
        a = torch.FloatTensor(self.memory.actions).to(device).unsqueeze(dim=1)
        r = torch.FloatTensor(self.memory.rewards).to(device).unsqueeze(dim=1)
        s_prime = torch.FloatTensor(self.memory.next_states).to(device)
        logp = torch.FloatTensor(self.memory.logprobs).to(device).\
            unsqueeze(dim=1)

        return s, a, r, s_prime, logp

    def train(self):
        s, a, r, s_prime, logp = self.make_batch()

        for i in range(nb_epoches):
            s_value, a_logprob = self.policy.evaluate(s, a)
            s_prime_value, _ = self.policy.evaluate(s_prime, None)

            td_target = r + gamma * s_prime_value
            delta = td_target - s_value
            delta = delta.detach().cpu().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lambda_ * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.FloatTensor(advantage_list).to(device)

            ratio = torch.exp(a_logprob - logp)

            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1 - CLIP_EPS,
                1 + CLIP_EPS
            ) * advantage

            loss = -torch.min(surr1, surr2) + \
                F.smooth_l1_loss(s_value, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


if __name__ == '__main__':
    env = ResampleEnv()
    model = PPO(state_dim=nb_bins * 4)

    for eid in range(nb_episodes):
        running_reward = 0
        print('Eposide {}'.format(eid))
        s = env.reset()
        for i in range(nb_sample_rounds):
            print('Round {}'.format(i))
            a, logp = model.select_action(s)
            s_prime, r = env.step(a)

            model.memory.states.append(s)
            model.memory.actions.append(a)
            model.memory.rewards.append(r)
            model.memory.next_states.append(s_prime)
            model.memory.logprobs.append(logp)

            s = s_prime
            running_reward += r

        model.train()
        model.memory.clear_mem()
        print('Reward in episode {}'.format(running_reward))
