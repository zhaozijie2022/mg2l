import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from mg2l.utils import pytorch_utils as ptu
from mg2l.utils.util import update_linear_schedule
import pickle

from .encoders import (PIAEncoder, SelfAttnEncoder, MLPEncoder, LocalEncoder,
                       RetNetEncoder, RNNEncoder, TransEncoder)


def make_ae(cfg):
    return ContextEncoder(cfg=cfg).to(ptu.device)


class BaseEncoder(nn.Module):
    def __init__(self, *kwargs):
        super(BaseEncoder, self).__init__()

    def prior(self, meta_batch):
        return torch.randn(meta_batch, self.n_agents, self.context_dim).to(ptu.device)

    def kl_div_normal(self, task_mu, task_log_std):
        task_std = F.softplus(task_log_std)
        return 0.5 * (task_mu ** 2 + task_std ** 2 - 2 * task_log_std - 1).mean()

    def kl_div_mutual(self, mu1, std1, mu2, std2):
        d = mu1.size(-1)
        det_std1 = std1.prod(dim=-1)
        det_std2 = std2.prod(dim=-1)
        inv_std1 = 1.0 / std1
        inv_std2 = 1.0 / std2
        tr_cov2_inv_cov1 = torch.einsum('...i,...i->...', inv_std2, inv_std1)
        mahalanobis_sq = ((mu2 - mu1) * inv_std2).pow(2).sum(dim=-1)
        kl_div = 0.5 * (tr_cov2_inv_cov1 + mahalanobis_sq - d + torch.log(det_std2 / det_std1))

        return kl_div


class ContextEncoder(BaseEncoder):
    def __init__(self, cfg):
        super(ContextEncoder, self).__init__()
        self.cfg = cfg
        self.obs_dim_n = cfg.obs_dim_n
        self.action_dim_n = cfg.action_dim_n
        self.n_agents = cfg.n_agents
        self.context_dim = cfg.context_dim
        self.n_tasks = cfg.n_tasks
        self.lr = cfg.lr
        self.kl_weight = cfg.kl_weight

        self.transition_encoders = [
            MLPEncoder(
                obs_dim=self.obs_dim_n[i],
                action_dim=self.action_dim_n[i],
                context_dim=self.context_dim,
                hidden_sizes=cfg.hidden_sizes_mlp,
            ).to(ptu.device)
            for i in range(self.n_agents)]
        self.temporal_encoder = PIAEncoder(context_dim=self.context_dim, is_agent=False).to(ptu.device)
        self.agent_encoder = PIAEncoder(context_dim=self.context_dim, is_agent=True).to(ptu.device)

        self.local_encoders = [
            LocalEncoder(
                context_dim=self.context_dim,
                hidden_sizes=cfg.hidden_sizes_mlp,
            ).to(ptu.device)
            for _ in range(self.n_agents)]
        self.global_log_std = nn.Linear(self.context_dim, self.context_dim, ).to(ptu.device)
        self.local_log_std = nn.Linear(self.context_dim, self.context_dim, ).to(ptu.device)

        encoder_parameters = []
        encoder_parameters.extend(self.temporal_encoder.parameters())
        encoder_parameters.extend(self.agent_encoder.parameters())
        for i in range(self.n_agents):
            encoder_parameters.extend(self.transition_encoders[i].parameters())
            encoder_parameters.extend(self.local_encoders[i].parameters())
        encoder_parameters.extend(self.global_log_std.parameters())
        encoder_parameters.extend(self.local_log_std.parameters())
        self.encoder_optimizer = AdamW(encoder_parameters, lr=self.lr)


    def transition_encode(self, obs_n, action_n, reward_n, next_obs_n):
        context_n = [ptu.zeros(0) for _ in range(self.n_agents)]
        for agent_id in range(self.n_agents):
            context_n[agent_id] = self.transition_encoders[agent_id](obs_n[agent_id], action_n[agent_id],
                                                                     reward_n[agent_id], next_obs_n[agent_id])
        return torch.stack(context_n, dim=-2)

    def temporal_encode(self, context_n):
        return self.temporal_encoder(context_n)

    def global_encode(self, context_ag):
        z = self.agent_encoder(context_ag)
        return z, self.global_log_std(z)

    def local_encode(self, context_ag):
        local_input = context_ag.detach().clone()
        z = [self.local_encoders[agent_id](local_input[:, agent_id, :]) for agent_id in range(self.n_agents)]
        z = torch.stack(z, dim=-2)
        return z, self.local_log_std(z)

    def forward(self, obs_n, action_n, reward_n, next_obs_n):
        context_n = self.transition_encode(obs_n, action_n, reward_n, next_obs_n)
        context_ag = self.temporal_encode(context_n)
        z_g, log_std_g = self.global_encode(context_ag)
        z_l, log_std_l = self.local_encode(context_ag)
        return z_g, log_std_g, z_l, log_std_l

    def encode(self, obs_n, action_n, reward_n, next_obs_n):
        z_g, log_std_g, z_l, log_std_l = self.forward(obs_n, action_n, reward_n, next_obs_n)
        return z_g + torch.randn_like(z_g) * F.softplus(log_std_g), \
               z_l + torch.randn_like(z_l) * F.softplus(log_std_l)

    def update(self, meta_tasks, data: Dict[str, torch.Tensor]):

        z_g, log_std_g, z_l, log_std_l = self.forward(*data["query"][:4])
        loss = self.kl_weight * (self.kl_div_normal(z_g, log_std_g) + self.kl_div_normal(z_l, log_std_l))
        z_g_q = z_g + torch.randn_like(z_g) * F.softplus(log_std_g)
        z_l_q = z_l + torch.randn_like(z_l) * F.softplus(log_std_l)
        update_info = {"kl_loss": loss.item()}

        key_pos, key_neg = data["key_pos"], data["key_neg"]
        key_pos = key_pos[:4]
        for i in range(4):
            key_neg[i] = key_neg[i].transpose(0, 1).reshape(self.n_agents, -1, *key_neg[i].shape[3:])
        z_g_p, z_l_p = self.encode(*key_pos)
        z_g_n, z_l_n = self.encode(*key_neg)

        z_g_n = z_g_n.reshape(z_g_p.shape[0], self.n_tasks - 1, -1)
        z_l_n = z_l_n.reshape(z_l_p.shape[0], self.n_tasks - 1, self.n_agents, -1)

        global_loss, _info = self.cl_loss(z_g_q, z_g_p, z_g_n)
        global_info = {}
        for key in _info:
            new_key = "global_" + key
            global_info[new_key] = _info[key]
        loss += global_loss
        update_info.update(global_info)

        local_loss, local_info = self.local_loss(z_g_q, z_g_p, z_g_n, z_l_q, z_l_p, z_l_n)
        loss += local_loss
        update_info.update(local_info)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        return update_info

    def cl_loss(self, z_q, z_p, z_n):
        b = z_q.shape[0]
        score_p = torch.bmm(z_q.view(b, 1, -1), z_p.view(b, -1, 1))
        score_n = torch.bmm(z_q.view(b, 1, -1), z_n.transpose(1, 2))
        y_hat = torch.cat([score_p.view(b, 1), score_n.view(b, self.n_tasks - 1)], dim=-1)

        cl_loss = F.cross_entropy(y_hat, ptu.zeros(b, dtype=torch.long))
        cl_acc = (y_hat.argmax(dim=-1) == 0).float().mean()
        return cl_loss, {"loss": cl_loss.item(), "acc": cl_acc}

    def local_loss(self, _z_g_q, _z_g_p, _z_g_n, z_l_q, z_l_p, z_l_n):
        z_g_q, z_g_p, z_g_n = _z_g_q.detach().clone(), _z_g_p.detach().clone(), _z_g_n.detach().clone()
        b = z_g_q.shape[0]
        z_g_q = z_g_q.repeat(self.n_agents, 1)
        z_g_p = z_g_p.repeat(self.n_agents, 1)
        z_l_q = z_l_q.reshape(b * self.n_agents, self.context_dim)
        z_l_p = z_l_p.reshape(b * self.n_agents, self.context_dim)
        z_g_n = z_g_n.repeat(self.n_agents, 1, 1)
        z_l_n = z_l_n.reshape(b * self.n_agents, self.n_tasks - 1, self.context_dim)

        z_q = torch.cat([z_g_q, z_l_q], dim=-1)
        z_p = torch.cat([z_g_p, z_l_p], dim=-1)
        z_n = torch.cat([z_g_n, z_l_n], dim=-1)

        loss, _info = self.cl_loss(z_q, z_p, z_n)
        local_info = {}
        for key in _info:
            new_key = "local_" + key
            local_info[new_key] = _info[key]

        return loss, local_info

    def local_kl_loss(self, _z_g, _log_std_g, z_l, log_std_l):
        z_g = _z_g.detach().clone()
        log_std_g = _log_std_g.detach().clone()

        std_g = F.softplus(log_std_g)
        std_l = F.softplus(log_std_l)

        z_g.repeat(self.n_agents, 1)
        std_g.repeat(self.n_agents, 1)

        z_l = z_l.reshape(-1, self.context_dim)
        std_l = std_l.reshape(-1, self.context_dim)

        loss = self.kl_div_mutual(z_g, std_g, z_l, std_l).mean()

        return loss, {"local_kl_loss": loss.item()}


    def lr_decay(self, epoch, total_epoch):
        update_linear_schedule(self.encoder_optimizer, epoch, total_epoch, self.lr)

    def save_model(self, save_path):
        models = {
            "transition_encoders": self.transition_encoders,
            "temporal_encoder": self.temporal_encoder,
            "agent_encoder": self.agent_encoder,
            "local_encoders": self.local_encoders,
        }
        save_path = os.path.join(save_path, "task_encoder.pth")
        with open(save_path, "wb") as f:
            torch.save(models, f)

    def load_model(self, load_path):
        load_path = os.path.join(load_path, "task_encoder.pth")
        with open(load_path, "rb") as f:
            models = torch.load(f, map_location=ptu.device)
        self.transition_encoders = [encoder.to(ptu.device) for encoder in models["transition_encoders"]]
        self.temporal_encoder = models["temporal_encoder"].to(ptu.device)
        self.agent_encoder = models["agent_encoder"].to(ptu.device)
        self.local_encoders = [encoder.to(ptu.device) for encoder in models["local_encoders"]]
