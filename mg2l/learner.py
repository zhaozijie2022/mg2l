import datetime
import json
import os
import time
import copy
from argparse import Namespace
from itertools import chain
from typing import Dict

import gym
import numpy as np
import torch
from gym.spaces import Box
from omegaconf import DictConfig, OmegaConf

import wandb
from ae.make_ae import make_ae
from algos.make_algo import make_algo
from buffer.make_buffer import make_buffer
from envs.make_env import make_env
from utils import util as utl, pytorch_utils as ptu


class Learner:
    def __init__(self, cfg: DictConfig):
        self.cfg = Namespace(**OmegaConf.to_container(cfg, resolve=True))
        utl.seed(self.cfg.seed)

        self.envs = make_env(cfg=self.cfg)
        self.envs.reset_task(0)
        self.n_agents = self.cfg.n_agents
        self.n_tasks = self.cfg.n_tasks
        self.max_ep_len = self.cfg.max_ep_len
        self.n_rollout_threads = self.cfg.n_rollout_threads
        self.obs_dim_n = [self.envs.observation_space[i].shape[0] for i in range(self.n_agents)]
        self.action_dim_n = [self.envs.action_space[i].n if isinstance(self.envs.action_space[i], gym.spaces.Discrete)
                             else self.envs.action_space[i].shape[0] for i in range(self.n_agents)]
        self.cfg.obs_dim_n = self.obs_dim_n
        self.cfg.action_dim_n = self.action_dim_n
        self.context_dim = self.cfg.context_dim
        self.task_idxes = list(range(self.cfg.n_tasks))
        print("initial envs: %s, done" % cfg.save_name)

        self.use_centralized_V = self.cfg.use_centralized_V
        self.use_obs_instead_of_state = self.cfg.use_obs_instead_of_state
        self.algo_hidden_size = self.cfg.algo_hidden_size
        self.recurrent_N = self.cfg.recurrent_N
        self.cfg.observation_space = self.envs.observation_space[:]
        self.cfg.action_space = self.envs.action_space[:]
        if self.use_centralized_V:
            self.cfg.share_observation_space = [
                Box(low=-np.inf, high=+np.inf, shape=(sum(self.cfg.obs_dim_n),), dtype=np.float32)
                for _ in range(self.cfg.n_agents)]
        self.agents = make_algo(cfg=self.cfg)
        print("initial algorithm: %s, done" % cfg.algo_name)

        self.task_encoder = make_ae(cfg=self.cfg)
        print("initial task encoder, done")

        self.ae_buffer_size = self.cfg.ae_buffer_size
        self.sample_ae_interval = self.cfg.sample_ae_interval
        self.rl_buffer, self.ae_buffer = make_buffer(self.cfg)
        tmp_cfg = copy.deepcopy(self.cfg)
        tmp_cfg.n_rollout_threads = self.cfg.n_ae_rollout_threads
        self.dummy_envs = make_env(tmp_cfg)
        self.dummy_buffer, _ = make_buffer(tmp_cfg)
        print("initial rl/task_encoder buffer, done")

        self.use_linear_lr_decay = self.cfg.use_linear_lr_decay
        self.n_iters = self.cfg.n_iters
        self.n_init_rollouts = self.cfg.n_init_rollouts
        self.n_rollout_threads = self.cfg.n_rollout_threads
        self.n_ae_rollout_threads = self.cfg.n_ae_rollout_threads
        self.ae_batch_size = self.cfg.ae_batch_size
        self.ae_updates_per_iter = self.cfg.ae_updates_per_iter
        self.eval_interval = self.cfg.eval_interval

        self.is_save_model = self.cfg.save_model
        self.save_interval = self.cfg.save_interval
        if self.cfg.load_model:
            self.load_model(self.cfg.load_model_path)
            print("!!!!!Note: Load model, done!!!!!")

        config_json = self.init_config()

        self.is_log_wandb = self.cfg.log_wandb
        self.log_interval = self.cfg.log_interval
        if self.is_log_wandb:
            wandb.init(project=self.cfg.save_name,
                       group=cfg.algo_name,
                       name=self.expt_name,
                       config=config_json, )
        print("initial learner, done")
        self._start_time = time.time()
        self._check_time = time.time()

    def train(self):

        for i in range(self.n_init_rollouts):
            ae_meta_tasks = np.random.choice(self.task_idxes, self.n_ae_rollout_threads, replace=True)
            self.rollout(ae_meta_tasks, self.dummy_buffer, self.dummy_envs, True)
        print("Initial task_encoder buffer, done. time: %.2f" % (time.time() - self._check_time))
        self._check_time = time.time()

        for iter_ in range(self.n_iters):
            if self.use_linear_lr_decay:
                for i in range(self.n_agents):
                    self.agents[i].policy.lr_decay(iter_, self.n_iters * 0.95)
                self.task_encoder.lr_decay(iter_, self.n_iters * 0.95)

            meta_tasks = np.random.choice(self.task_idxes, self.n_rollout_threads, replace=True)
            rollout_info = self.rollout(meta_tasks, self.rl_buffer, self.envs, False)
            rl_train_info = self.rl_update(meta_tasks)

            if (iter_ + 1) % self.cfg.train_ae_interval == 0:
                ae_train_info = self.ae_update(meta_tasks)
            else:
                ae_train_info = {}

            if (iter_ + 1) % self.log_interval == 0:
                self.log(iter_ + 1, meta_tasks,
                         rollout_info=rollout_info,
                         ae_train_info=ae_train_info,
                         rl_train_info=rl_train_info)
                print([self.ae_buffer.task_buffers[task_idx].size for task_idx in self.task_idxes])

            if self.is_save_model and (iter_ + 1) % self.save_interval == 0:
                save_path = os.path.join(self.output_path, 'models_%d.pt' % (iter_ + 1))
                if self.is_save_model:
                    os.makedirs(save_path, exist_ok=True)
                    self.save_model(save_path)
                    print("model saved in %s" % save_path)

            if (iter_ + 1) % self.sample_ae_interval == 0:
                ae_meta_tasks = np.random.choice(self.task_idxes, self.n_ae_rollout_threads, replace=True)
                self.rollout(ae_meta_tasks, self.dummy_buffer, self.dummy_envs, True)

        if self.is_log_wandb:
            wandb.finish()
            print("wandb run has finished")
            print("")

        self.envs.close()
        self.dummy_envs.close()
        print("multi processing envs have been closed")
        print("")

    def rollout(self, meta_tasks, r_buffer, r_envs, is_ae_buffer=False):
        meta_batch = len(meta_tasks)
        assert r_envs.n_envs == meta_batch
        assert r_buffer[0].n_rollout_threads == meta_batch

        _rew, _sr = 0., 0.
        _rew_p = np.zeros((meta_batch, 1))
        self.warmup(meta_tasks, r_buffer, r_envs)
        _done = np.zeros((self.n_agents, self.max_ep_len, meta_batch), dtype=np.float32)
        _actions_ae = np.zeros((self.n_agents, self.max_ep_len, meta_batch, self.action_dim_n[0]), dtype=np.float32)

        for cur_step in range(self.max_ep_len):
            with torch.no_grad():
                if not is_ae_buffer:
                    data4z = self.sample_batch_ae(meta_tasks, self.ae_batch_size)
                    z_critic, z_actor = self.task_encoder.encode(*data4z)
                    z_actor = z_actor.permute(1, 0, 2)
                    z_critic = z_critic.unsqueeze(0). repeat(self.n_agents, 1, 1)
                else:
                    z_actor = ptu.randn(self.n_agents, self.n_ae_rollout_threads, self.context_dim)
                    z_critic = ptu.randn(self.n_ae_rollout_threads, self.context_dim).unsqueeze(0). repeat(self.n_agents, 1, 1)

            (values, actions, action_log_probs, rnn_states,
             rnn_states_critic, actions_ae) = self.collect(cur_step, r_buffer, z_actor, z_critic)
            obs, rewards, dones, infos = r_envs.step(actions)
            data = (obs, rewards, dones, infos, values, actions,
                    action_log_probs, rnn_states, rnn_states_critic,)
            self.insert(data, r_buffer)

            _done[:, cur_step, :] = np.transpose(dones, (1, 0)).copy()
            _actions_ae[:, cur_step, :, :] = np.transpose(actions_ae, (1, 0, 2)).copy()
            _rew += np.mean(rewards)

        if is_ae_buffer:
            for i_t, task_idx in enumerate(meta_tasks):
                self.ae_buffer.add_samples(
                    task_idx=task_idx,
                    n_obs_n=[r_buffer[agent_id].obs[:-1, i_t, :] for agent_id in range(self.n_agents)],
                    n_action_n=_actions_ae[:, :, i_t, :],
                    n_reward_n=[r_buffer[agent_id].rewards[:, i_t, :] for agent_id in range(self.n_agents)],
                    n_next_obs_n=[r_buffer[agent_id].obs[1:, i_t, :] for agent_id in range(self.n_agents)],
                    n_done_n=_done[:, :, i_t],
                )
        else:
            data4z = self.sample_batch_ae(meta_tasks, self.ae_batch_size)
            z_g, _ = self.task_encoder.encode(*data4z)
            self.compute(r_buffer, z_g)
        return {"reward": _rew}

    def warmup(self, meta_tasks, r_buffer, r_envs):
        r_envs.meta_reset_task(meta_tasks)
        obs = r_envs.reset()
        share_obs = obs.reshape(obs.shape[0], -1).copy()

        for agent_id in range(self.n_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            r_buffer[agent_id].share_obs[0] = share_obs.copy()
            r_buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, cur_step, r_buffer, z_actor, z_critic):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.n_agents):
            self.agents[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = \
                self.agents[agent_id].policy.get_actions(
                    cent_obs=r_buffer[agent_id].share_obs[cur_step],
                    obs=r_buffer[agent_id].obs[cur_step],
                    rnn_states_actor=r_buffer[agent_id].rnn_states[cur_step],
                    rnn_states_critic=r_buffer[agent_id].rnn_states_critic[cur_step],
                    masks=r_buffer[agent_id].masks[cur_step],
                    z_actor=z_actor[agent_id],
                    z_critic=z_critic[agent_id],)
            values.append(ptu.get_numpy(value))
            action = ptu.get_numpy(action)
            actions.append(action)
            action_log_probs.append(ptu.get_numpy(action_log_prob))
            rnn_states.append(ptu.get_numpy(rnn_state))
            rnn_states_critic.append(ptu.get_numpy(rnn_state_critic))

        values = np.array(values).transpose((1, 0, 2))
        actions = np.array(actions).transpose((1, 0, 2))
        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_ae = np.eye(self.envs.action_space[0].n)[actions.reshape(-1)].reshape(*actions.shape[:2], -1)
        else:
            actions_ae = actions.copy()
        action_log_probs = np.array(action_log_probs).transpose((1, 0, 2))
        rnn_states = np.array(rnn_states).transpose((1, 0, 2, 3))
        rnn_states_critic = np.array(rnn_states_critic).transpose((1, 0, 2, 3))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_ae

    def insert(self, data, r_buffer):
        (obs, rewards, dones, infos, values, actions,
         action_log_probs, rnn_states, rnn_states_critic,) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.algo_hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.algo_hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((r_buffer[0].n_rollout_threads, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.n_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            r_buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def compute(self, r_buffer, z):
        for agent_id in range(self.n_agents):
            self.agents[agent_id].prep_rollout()
            next_value = self.agents[agent_id].policy.get_values(
                cent_obs=r_buffer[agent_id].share_obs[-1],
                context=z,
                rnn_states_critic=r_buffer[agent_id].rnn_states_critic[-1],
                masks=r_buffer[agent_id].masks[-1],
            )
            next_value = ptu.get_numpy(next_value)
            r_buffer[agent_id].compute_returns(next_value, self.agents[agent_id].value_normalizer)


    def sample_tasks(self, rew_tasks, meta_batch, is_rand=True):
        assert len(rew_tasks) == self.n_tasks
        if not is_rand:
            return np.random.choice(self.task_idxes, meta_batch, replace=True)

        x = (rew_tasks - np.mean(rew_tasks)) / np.std(rew_tasks)

        x = np.clip(x, -0.5, 0.5)
        probs = np.exp(-x) / np.sum(np.exp(-x))
        cum_probs = np.cumsum(probs)
        uniform_rands = np.random.rand(meta_batch)
        return list(np.searchsorted(cum_probs, uniform_rands))

    def sample_batch_ae(self, meta_tasks, batch_size):
        batch = []
        for task_idx in meta_tasks:
            batch.append(self.ae_buffer.random_batch(task_idx=task_idx, batch_size=batch_size))
        obs_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["obs"]) for tmp2 in tmp1]) for tmp1 in batch], dim=1)
        action_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["action"]) for tmp2 in tmp1]) for tmp1 in batch], dim=1)
        reward_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["reward"]) for tmp2 in tmp1]) for tmp1 in batch], dim=1)
        next_obs_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["next_obs"]) for tmp2 in tmp1]) for tmp1 in batch], dim=1)
        return [obs_n, action_n, reward_n, next_obs_n]

    def rl_update(self, meta_tasks):
        data4z = self.sample_batch_ae(meta_tasks, self.ae_batch_size)

        update_info = {}
        count = 0
        for agent_id in range(self.n_agents):
            self.agents[agent_id].prep_training()
            agent_train_info = self.agents[agent_id].train(
                buffer=self.rl_buffer[agent_id],
                update_actor=True,
                ae=self.task_encoder,
                data4z=data4z,
            )
            self.rl_buffer[agent_id].after_update()

            count += 1
            for key, value in agent_train_info.items():
                if key not in update_info:
                    update_info[key] = value
                else:
                    update_info[key] += value
        for key, value in update_info.items():
            update_info[key] = value / count
        return update_info

    def ae_update(self, meta_tasks):
        meta_batch = len(meta_tasks)
        update_info = {}
        num_iters = self.ae_updates_per_iter
        for r in range(num_iters):
            data = {"query": self.sample_batch_ae(meta_tasks, self.ae_batch_size),
                    "key_pos": self.sample_batch_ae(meta_tasks, self.ae_batch_size)}
            obs_neg, action_neg, reward_neg, next_obs_neg = [[[] for _ in meta_tasks] for _ in range(4)]
            for i in range(meta_batch):
                neg_meta_batch = self.task_idxes[:meta_tasks[i]] + self.task_idxes[meta_tasks[i] + 1:]
                obs_neg[i], action_neg[i], reward_neg[i], next_obs_neg[i] = \
                    self.sample_batch_ae(neg_meta_batch, self.ae_batch_size)
            data["key_neg"] = [torch.stack(obs_neg), torch.stack(action_neg),
                               torch.stack(reward_neg), torch.stack(next_obs_neg)]

            ae_losses = self.task_encoder.update(meta_tasks=meta_tasks, data=data)
            for key, value in ae_losses.items():
                if key in update_info:
                    update_info[key] += value / num_iters
                else:
                    update_info[key] = value / num_iters
        return update_info


    def log(self, iter_, meta_tasks, **kwargs):
        if self.is_log_wandb:
            for key, value in kwargs.items():
                wandb.log(value, step=iter_)

        print("")
        print("******** iter: %d, iter_time: %.2fs, total_time: %.2fs" %
              (iter_, time.time() - self._check_time, time.time() - self._start_time))
        print("meta_tasks: ", meta_tasks)
        for key, value in kwargs.items():
            print("%s" % key + "".join([", %s: %.4f" % (k, v) for k, v in value.items()]))
        self._check_time = time.time()

    def init_config(self):
        date_dir = datetime.datetime.now().strftime("%m%d%H%M_")
        seed_dir = 'sd%d' % self.cfg.seed
        self.expt_name = date_dir + seed_dir
        self.output_path = str(os.path.join(self.cfg.main_save_path, self.cfg.save_name, self.expt_name))
        self.cfg.output_path = self.output_path
        config_json = vars(self.cfg)

        config_json.pop("action_space")
        config_json.pop("observation_space")
        config_json.pop("share_observation_space")

        if self.is_save_model:
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, "config.json"), 'w') as f:
                json.dump(config_json, f, indent=4)
        return config_json

    def save_model(self, save_path):
        for agent_id in range(self.n_agents):
            self.agents[agent_id].save_model(save_path)
        self.task_encoder.save_model(save_path)

    def load_model(self, load_path):
        for agent_id in range(self.n_agents):
            self.agents[agent_id].load_model(load_path)
        self.task_encoder.load_model(load_path)
