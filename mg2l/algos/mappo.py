import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mg2l.utils.util import get_gard_norm, huber_loss, mse_loss
from mg2l.utils.valuenorm import ValueNorm
from mg2l.utils.util import check, update_linear_schedule
import mg2l.utils.pytorch_utils as ptu
import pickle

import torch
from mg2l.algos.r_actor_critic import R_Actor, R_Critic
from mg2l.utils.util import update_linear_schedule
import os


class MAPPOPolicy:
    def __init__(self, cfg, obs_space, cent_obs_space, act_space, agent_id):
        self.device = ptu.device
        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.opti_eps = cfg.opti_eps
        self.weight_decay = cfg.weight_decay
        self.agent_id = agent_id

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(cfg, self.obs_space, self.act_space, ptu.device)
        self.critic = R_Critic(cfg, self.share_obs_space, ptu.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr,
                                                eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.actor_lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                    masks, z_actor, z_critic, available_actions=None, deterministic=False):
        obs = ptu.from_numpy(obs)
        x_actor = torch.cat([obs, z_actor], dim=-1)
        actions, action_log_probs, rnn_states_actor = self.actor(x_actor, rnn_states_actor, masks,
                                                                 available_actions, deterministic)

        cent_obs = ptu.from_numpy(cent_obs)
        x_critic = torch.cat([cent_obs, z_critic], dim=-1)
        values, rnn_states_critic = self.critic(x_critic, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, context, rnn_states_critic, masks):
        cent_obs = ptu.from_numpy(cent_obs)
        x_critic = torch.cat([cent_obs, context], dim=-1)
        values, _ = self.critic(x_critic, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                         action, masks, available_actions=None, active_masks=None, z_g=None, z_l=None):
        obs = ptu.from_numpy(obs)
        z_actor = z_l[:, self.agent_id, :].detach().clone()
        x_actor = torch.cat([obs, z_actor], dim=-1)
        action_log_probs, dist_entropy = self.actor.evaluate_actions(x_actor, rnn_states_actor, action,
                                                                     masks, available_actions, active_masks)

        cent_obs = ptu.from_numpy(cent_obs)
        x_critic = torch.cat([cent_obs, z_g], dim=-1)
        values, _ = self.critic(x_critic, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


class MAPPOTrainer:
    def __init__(self, cfg, policy, agent_id):
        self.tpdv = dict(dtype=torch.float32, device=ptu.device)
        self.policy = policy
        self.agent_id = agent_id

        self.clip_param = cfg.clip_param
        self.ppo_epoch = cfg.ppo_epoch
        self.num_mini_batch = cfg.num_mini_batch
        self.data_chunk_length = cfg.data_chunk_length
        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.huber_delta = cfg.huber_delta

        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._use_naive_recurrent = cfg.use_naive_recurrent_policy
        self._use_max_grad_norm = cfg.use_max_grad_norm
        self._use_clipped_value_loss = cfg.use_clipped_value_loss
        self._use_huber_loss = cfg.use_huber_loss
        self._use_popart = cfg.use_popart
        self._use_value_norm = cfg.use_valuenorm
        self._use_value_active_masks = cfg.use_value_active_masks
        self._use_policy_active_masks = cfg.use_policy_active_masks

        assert (self._use_popart and self._use_value_norm) is False, (
            "self._use_popart and self._use_value_norm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_value_norm:
            self.value_normalizer = ValueNorm(1, device=ptu.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_value_norm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True, ae=None, data4z=None):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, \
            actions_batch, value_preds_batch, return_batch, masks_batch, \
            active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch,\
            indices = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        z_g, z_l = ae.encode(*data4z)
        z_g = z_g.unsqueeze(0).repeat(len(indices), 1, 1).reshape(-1, *z_g.shape[1:])
        z_g = z_g[indices]
        z_l = z_l.unsqueeze(0).repeat(len(indices), 1, 1, 1).reshape(-1, *z_l.shape[1:])
        z_l = z_l[indices]


        values, action_log_probs, dist_entropy = \
            self.policy.evaluate_actions(share_obs_batch, obs_batch, rnn_states_batch,
                                         rnn_states_critic_batch, actions_batch, masks_batch,
                                         available_actions_batch, active_masks_batch,
                                         z_g, z_l)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        ae.encoder_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()
        ae.encoder_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True, ae=None, data4z=None):
        if self._use_popart or self._use_value_norm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {'value_loss': 0, 'policy_loss': 0, 'dist_entropy': 0,
                      'actor_grad_norm': 0, 'critic_grad_norm': 0, 'ratio': 0}

        for epoch in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor, ae, data4z)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def save_model(self, save_path):
        save_path = os.path.join(save_path, "agent_%d.pth" % self.agent_id)

        with open(save_path, 'wb') as f:
            torch.save(self.policy, f)

    def load_model(self, load_path):
        load_path = os.path.join(load_path, "agent_%d.pth" % self.agent_id)
        with open(load_path, 'rb') as f:
            self.policy = torch.load(f, map_location=ptu.device)
            self.policy.device = ptu.device
            self.policy.actor.to(ptu.device)
            self.policy.actor.device = ptu.device
            self.policy.actor.tpdv = dict(dtype=torch.float32, device=ptu.device)
            self.policy.critic.to(ptu.device)
            self.policy.critic.device = ptu.device
            self.policy.critic.tpdv = dict(dtype=torch.float32, device=ptu.device)
