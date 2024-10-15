import pickle
from typing import List

import numpy as np


class MultiTaskReplayBuffer:
    def __init__(
            self,
            max_size,
            obs_dim_n: List[int],
            action_dim_n: List[int],
            max_episode_steps,
            tasks: List[int],
            n_agents,
            **kwargs,
    ):
        self.max_size = max_size
        self.obs_dim_n = obs_dim_n
        self.action_dim_n = action_dim_n
        self.max_episode_steps = max_episode_steps
        self.tasks = tasks
        self.n_agents = n_agents
        self.task_buffers = dict([(idx, MultiAgentReplayBuffer(
            max_size=max_size,
            obs_dim_n=obs_dim_n,
            action_dim_n=action_dim_n,
            max_episode_steps=max_episode_steps,
            n_agents=n_agents,
            **kwargs
        )) for idx in self.tasks])

    def clear(self):
        for task_idx in self.tasks:
            self.task_buffers[task_idx].clear()

    def add_sample(self, task_idx, obs_n: List[np.ndarray],
                   action_n, reward_n, next_obs_n, done_n, ):
        self.task_buffers[task_idx].add_sample(obs_n=obs_n, action_n=action_n, reward_n=reward_n,
                                               next_obs_n=next_obs_n, done_n=done_n)

    def add_samples(self, task_idx, n_obs_n: List[List[np.ndarray]],
                    n_action_n, n_reward_n, n_next_obs_n, n_done_n):
        self.task_buffers[task_idx].add_samples(n_obs_n=n_obs_n, n_action_n=n_action_n, n_reward_n=n_reward_n,
                                                n_next_obs_n=n_next_obs_n, n_done_n=n_done_n)

    def add_episode(self, task_idx, ep_obs_n: List[List[np.ndarray]],
                    ep_action_n, ep_reward_n, ep_next_obs_n, ep_done_n):
        self.task_buffers[task_idx].add_episode(ep_obs_n=ep_obs_n, ep_action_n=ep_action_n, ep_reward_n=ep_reward_n,
                                                ep_next_obs_n=ep_next_obs_n, ep_done_n=ep_done_n)

    def sample_data(self, task_idx, indices):
        return self.task_buffers[task_idx].sample_data(indices)

    def random_batch(self, task_idx, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task_idx].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task_idx].random_batch(batch_size)
        return batch

    def random_episodes(self, task_idx, n_episodes):
        return self.task_buffers[task_idx].random_episodes(n_episodes)

    def can_sample_batch(self, task_idx, batch_size):
        return self.task_buffers[task_idx].can_sample_batch(batch_size)

    def can_sample_episodes(self, task_idx, n_episodes):
        return self.task_buffers[task_idx].can_sample_episodes(n_episodes)

    def num_steps_can_sample(self, task_idx):
        return self.task_buffers[task_idx].num_steps_can_sample()

    def add_path(self, task_idx, path):
        pass

    def add_paths(self, task_idx, paths):
        pass

    def clear_buffer(self, task_idx):
        self.task_buffers[task_idx].clear()

    def num_complete_episodes(self, task_idx):
        return self.task_buffers[task_idx].num_complete_episodes()

    def save_buffer(self, task_idx, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.task_buffers[task_idx], f)


class MultiAgentReplayBuffer:
    def __init__(
            self,
            max_size,
            obs_dim_n: List[int],
            action_dim_n: List[int],
            max_episode_steps,
            n_agents,
            **kwargs,
    ):

        self.max_size = max_size
        self.obs_dim_n = obs_dim_n
        self.action_dim_n = action_dim_n
        self.max_episode_steps = max_episode_steps
        self.n_agents = n_agents
        self.agent_buffers = dict([(idx, SimpleReplayBuffer(
            max_size=max_size,
            obs_dim=obs_dim_n[idx],
            action_dim=action_dim_n[idx],
            max_episode_steps=max_episode_steps,
            **kwargs
        )) for idx in range(self.n_agents)])

        self._top = 0
        self.size = 0
        self.episode_indices = []

    def clear(self):
        self._top = 0
        self.size = 0
        self.episode_indices = []
        for agent_id in range(self.n_agents):
            self.agent_buffers[agent_id].clear()

    def add_sample(self, obs_n: List[np.ndarray],
                   action_n, reward_n, next_obs_n, done_n,):
        for agent_id in range(self.n_agents):
            self.agent_buffers[agent_id].add_sample(obs=obs_n[agent_id],
                                                    action=action_n[agent_id],
                                                    reward=reward_n[agent_id],
                                                    next_obs=next_obs_n[agent_id],
                                                    done=done_n[agent_id])
        self._top = (self._top + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def add_samples(self, n_obs_n: List[List[np.ndarray]],
                    n_action_n, n_reward_n, n_next_obs_n, n_done_n,):
        n_samples = len(n_obs_n[0])

        n_obs_n_ = [np.array(n_obs_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_action_n_ = [np.array(n_action_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_reward_n_ = [np.array(n_reward_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_next_obs_n_ = [np.array(n_next_obs_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_done_n_ = [np.array(n_done_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]

        for agent_id in range(self.n_agents):
            self.agent_buffers[agent_id].add_samples(n_obs=n_obs_n_[agent_id],
                                                     n_action=n_action_n_[agent_id],
                                                     n_reward=n_reward_n_[agent_id],
                                                     n_next_obs=n_next_obs_n_[agent_id],
                                                     n_done=n_done_n_[agent_id],)

        self._top = (self._top + n_samples) % self.max_size
        self.size = self.agent_buffers[0].size

    def add_episode(self, ep_obs_n, ep_action_n, ep_reward_n, ep_next_obs_n, ep_done_n):
        n_samples = len(ep_obs_n[0])
        if self._top + n_samples <= self.max_size:
            self.episode_indices.append(list(range(self._top, self._top + n_samples)))
            self.add_samples(ep_obs_n, ep_action_n, ep_reward_n, ep_next_obs_n, ep_done_n)
        else:
            self.episode_indices.append([])
            for i in range(n_samples):
                self.episode_indices[-1].append(self._top)
                trans_obs_n = [ep_obs_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_action_n = [ep_action_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_reward_n = [ep_reward_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_next_obs_n = [ep_next_obs_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_done_n = [ep_done_n[i][agent_id] for agent_id in range(self.n_agents)]
                self.add_sample(trans_obs_n, trans_action_n, trans_reward_n, trans_next_obs_n, trans_done_n)

        if self.size == self.max_size:
            while set(self.episode_indices[0]) == set(range(self.max_episode_steps)):
                del self.episode_indices[0]

    def sample_data(self, indices):
        return [self.agent_buffers[agent_id].sample_data(indices)
                for agent_id in range(self.n_agents)]

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        indices = []
        while len(indices) < batch_size:
            start = np.random.randint(low=0, high=len(self.episode_indices))
            indices += self.episode_indices[start]
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def random_episodes(self, n_episodes):
        sampled_data = []
        for ep in range(n_episodes):
            start = np.random.randint(low=0, high=len(self.episode_indices))
            sampled_data.append(self.sample_data(self.episode_indices[start]))
        return sampled_data


    def can_sample_batch(self, batch_size):
        return batch_size <= self.size

    def can_sample_episodes(self, n_episodes=0):
        return len(self.episode_indices) >= n_episodes

    def num_steps_can_sample(self):
        return self.size

    def num_complete_episodes(self):
        return len(self.episode_indices)


class SimpleReplayBuffer:
    def __init__(
            self,
            max_size,
            obs_dim,
            action_dim,
            max_episode_steps,
            **kwargs,
    ):
        self.max_size = max_size
        self.observation_dim = obs_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps

        self._obs = np.zeros((max_size, obs_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._next_obs = np.zeros((max_size, obs_dim))
        self._done = np.zeros((max_size, 1))

        self._top = 0
        self.size = 0
        self.episode_indices = []

    def clear(self):
        self._top = 0
        self.size = 0
        self.episode_indices = []

    def add_sample(self, obs, action, reward, next_obs, done):
        self._obs[self._top] = obs
        self._actions[self._top] = action
        self._reward[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._done[self._top] = done

        self._top = (self._top + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def add_samples(self, n_obs, n_action, n_reward, n_next_obs, n_done, **kwargs):
        n_samples = n_obs.shape[0]
        if self._top + n_samples <= self.max_size:
            self._obs[self._top: self._top + n_samples] = n_obs
            self._actions[self._top: self._top + n_samples] = n_action
            self._reward[self._top: self._top + n_samples] = n_reward
            self._next_obs[self._top: self._top + n_samples] = n_next_obs
            self._done[self._top: self._top + n_samples] = n_done

            self._top = (self._top + n_samples) % self.max_size
            self.size = min(self.size + n_samples, self.max_size)
        else:
            for i in range(n_samples):
                self.add_sample(n_obs[i], n_action[i], n_reward[i], n_next_obs[i], n_done[i])

    def add_episode(self, ep_obs, ep_action, ep_reward, ep_next_obs, ep_done):
        if self._top + ep_obs.shape[0] <= self.max_size:
            self.episode_indices.append(list(range(self._top, self._top + ep_obs.shape[0])))
            self.add_samples(ep_obs, ep_action, ep_reward, ep_next_obs, ep_done)
        else:
            self.episode_indices.append([])
            for i in range(ep_obs.shape[0]):
                self.episode_indices[-1].append(self._top)
                self.add_sample(ep_obs[i], ep_action[i], ep_reward[i], ep_next_obs[i], ep_done[i])
        if self.size == self.max_size:
            while set(self.episode_indices[0]) & set(self.episode_indices[-1]):
                del self.episode_indices[0]

    def sample_data(self, indices):
        return dict(
            obs=self._obs[indices],
            action=self._actions[indices],
            reward=self._reward[indices],
            next_obs=self._next_obs[indices],
            done=self._done[indices],
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        indices = []
        while len(indices) < batch_size:
            start = np.random.randint(low=0, high=len(self.episode_indices))
            indices += self.episode_indices[start]
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def random_episodes(self, n_episodes):
        sampled_data = []
        for ep in range(n_episodes):
            start = np.random.randint(low=0, high=len(self.episode_indices))
            sampled_data.append(self.sample_data(self.episode_indices[start]))
        return sampled_data

    def can_sample_batch(self, batch_size):
        return batch_size <= self.size

    def can_sample_episodes(self, n_episodes=0):
        return len(self.episode_indices) >= n_episodes

    def num_steps_can_sample(self):
        return self.size

    def num_complete_episodes(self):
        return len(self.episode_indices)