import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from mg2l.utils import pytorch_utils as ptu
from mg2l.utils.networks import Mlp


class SelfAttnEncoder(nn.Module):
    def __init__(self, context_dim, n_heads, is_agent=True):
        super(SelfAttnEncoder, self).__init__()
        assert context_dim % n_heads == 0, "d_model should be divisible by n_heads"

        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = context_dim // n_heads

        self.W_q = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_k = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_v = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_o = ptu.m_init(nn.Linear(context_dim, context_dim))

        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()
        B, L, _ = x.size()

        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, L, self.context_dim)
        output = self.W_o(output)[:, 0]
        if not self.is_agent:
            output = output.view(meta_batch, batch_size, self.context_dim)
        return output


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_dim=1, done_dim=0,
                 hidden_sizes=[64], context_dim=32, normalize=False,
                 output_activation=ptu.identity, **kwargs):
        super(MLPEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.done_dim = done_dim
        self.use_done = True if done_dim else False
        self.latent_dim = context_dim

        self.encoder = Mlp(input_size=obs_dim * 2 + action_dim + reward_dim + done_dim,
                           output_size=context_dim,
                           hidden_sizes=hidden_sizes,
                           hidden_activation=F.gelu,
                           output_activation=output_activation, )

        self.normalize = normalize

    def forward(self, obs, action, reward, next_obs, done=None):
        if self.use_done:
            f_input = torch.cat([obs, action, reward, next_obs, done], dim=-1)
            out = self.encoder(f_input)
        else:
            f_input = torch.cat([obs, action, reward, next_obs], dim=-1)
            out = self.encoder(f_input)
        return F.normalize(out) if self.normalize else out


class LocalEncoder(nn.Module):
    def __init__(self, context_dim=32, hidden_sizes=[64], normalize=False,
                 output_activation=ptu.identity, **kwargs):
        super(LocalEncoder, self).__init__()

        self.latent_dim = context_dim

        self.encoder = Mlp(input_size=context_dim,
                           output_size=context_dim,
                           hidden_sizes=hidden_sizes,
                           hidden_activation=F.gelu,
                           output_activation=output_activation, )

        self.normalize = normalize

    def forward(self, z):
        out = self.encoder(z)
        return F.normalize(out) if self.normalize else out


class PIAEncoder(nn.Module):
    def __init__(self, context_dim, is_agent=True):
        super(PIAEncoder, self).__init__()
        self.context_dim = context_dim
        self.W_a = ptu.m_init(nn.Linear(context_dim, 1))
        self.W_v = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_o = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()

        attn = self.W_a(x).transpose(-1, -2)
        attn = F.softmax(attn, dim=-1)

        x = self.W_v(x)
        output = torch.matmul(attn, x).squeeze(1)
        output = self.W_o(output)
        if not self.is_agent:
            output = output.view(meta_batch, batch_size, self.context_dim)
        return output


class RNNEncoder(nn.Module):
    def __init__(self, context_dim, is_agent=True):
        super(RNNEncoder, self).__init__()
        self.context_dim = context_dim
        self.is_agent = is_agent

        self.fc1 = nn.Linear(context_dim, context_dim)
        self.rnn = nn.GRU(context_dim, context_dim, 1, batch_first=True)
        self.fc2 = nn.Linear(context_dim, context_dim)

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()

        h = F.gelu(self.fc1(x))
        o, new_h = self.rnn(h)
        o = F.gelu(self.fc2(o))
        o = o[:, -1, :]
        if not self.is_agent:
            o = o.view(meta_batch, batch_size, self.context_dim)
        return o

    def seq(self, x):
        return self.forward(x)

    def one_step(self, x, h):
        h = F.gelu(self.fc1(x))
        o, new_h = self.rnn(h, h)
        o = F.gelu(self.fc2(o))
        return o, new_h


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = ptu.m_init(nn.Linear(d_model, d_model))
        self.W_k = ptu.m_init(nn.Linear(d_model, d_model))
        self.W_v = ptu.m_init(nn.Linear(d_model, d_model))
        self.W_o = ptu.m_init(nn.Linear(d_model, d_model))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        position = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, num_hiddens, 2).float() * (-np.log(10000.0) / num_hiddens))
        self.P[:, :, 0::2] = torch.sin(position * div_term)
        self.P[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.P[:, :x.shape[1], :].to(x.device)


class EncodeBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncodeBlock, self).__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = Mlp(d_model, d_model, [d_model], hidden_activation=F.gelu)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class TransEncoder(nn.Module):
    def __init__(self, context_dim, n_blocks, n_heads, is_agent=True):
        super(TransEncoder, self).__init__()
        self.context_dim = context_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads

        self.pe = PositionalEncoding(context_dim)
        self.ln = nn.LayerNorm(context_dim)
        self.blocks = nn.Sequential(*[EncodeBlock(context_dim, n_heads) for _ in range(n_blocks)])
        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
            output = self.blocks(self.ln(self.pe(x)))[:, 0]
            return output.view(meta_batch, batch_size, self.context_dim)
        else:
            return self.blocks(self.ln(x))[:, 0]


class ComplexGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(ComplexGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(ptu.ones(num_channels))
        self.bias = nn.Parameter(ptu.zeros(num_channels))

    def forward(self, X):
        X = X.reshape(-1, self.num_groups, self.num_channels // self.num_groups)
        mean = X.mean(dim=2, keepdim=True)
        var = X.var(dim=2, keepdim=True)
        X = (X - mean) / torch.sqrt(var + self.eps)
        X = X.reshape(-1, self.num_channels)
        X = X * self.weight + self.bias

        return X


class ComplexLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ComplexLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(ptu.ones(num_channels))
        self.bias = nn.Parameter(ptu.zeros(num_channels))

    def forward(self, X):
        X_shape = X.shape
        X = X.reshape(-1, X_shape[-1])
        mean = X.mean(dim=1, keepdim=True)
        var = X.abs().var(dim=1, keepdim=True)
        X = (X - mean) / torch.sqrt(var + self.eps)
        X = X * self.weight + self.bias
        X = X.reshape(X_shape)
        return X


class ComplexFFN(nn.Module):

    def __init__(self, hidden_size, ffn_size):
        super(ComplexFFN, self).__init__()
        self.W1 = nn.Parameter(ptu.randn(hidden_size, ffn_size) / math.sqrt(hidden_size))
        self.W2 = nn.Parameter(ptu.randn(ffn_size, hidden_size) / math.sqrt(ffn_size))
        self.gelu = lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, X):
        X = X @ self.W1.to(X)
        X = self.gelu(X)
        X = X @ self.W2.to(X)

        return X


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, precision="single"):
        super(SimpleRetention, self).__init__()

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32

        self.precision = precision
        self.hidden_size = hidden_size
        self.gamma = gamma

        self.i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

        self.W_Q = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_K = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_V = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)

        self.theta = ptu.randn(hidden_size) / hidden_size
        self.theta = nn.Parameter(self.theta)

    def forward(self, X):
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        if X.dtype != self.complex_type:
            X = torch.complex(X, ptu.zeros_like(X)).to(self.complex_type)

        i = self.i.to(X.device)
        ns = torch.arange(1, sequence_length + 1, dtype=self.real_type, device=X.device)
        ns = torch.complex(ns, ptu.zeros_like(ns)).to(self.complex_type)
        Theta = []
        for n in ns:
            Theta.append(torch.exp(i * n * self.theta))
        Theta = torch.stack(Theta, dim=0)
        Theta_bar = Theta.conj()
        Q = (X @ self.W_Q.to(self.complex_type)) * Theta.unsqueeze(0)
        K = (X @ self.W_K.to(self.complex_type)) * Theta_bar.unsqueeze(0)
        V = X @ self.W_V.to(self.complex_type)
        att = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return att @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        if x_n.dtype != self.complex_type:
            x_n = torch.complex(x_n, ptu.zeros_like(x_n)).to(self.complex_type)

        n = torch.tensor(n, dtype=self.complex_type, device=x_n.device)

        Theta = torch.exp(self.i * n * self.theta)
        Theta_bar = Theta.conj()

        Q = (x_n @ self.W_Q.to(self.complex_type)) * Theta
        K = (x_n @ self.W_K.to(self.complex_type)) * Theta_bar
        V = x_n @ self.W_V.to(self.complex_type)

        s_n = self.gamma * s_n_1 + K.unsqueeze(2) @ V.unsqueeze(1)

        return (Q.unsqueeze(1) @ s_n).squeeze(1), s_n

    def _get_D(self, sequence_length):
        D = ptu.zeros((sequence_length, sequence_length), dtype=self.real_type, requires_grad=False)
        for n in range(sequence_length):
            for m in range(sequence_length):
                if n >= m:
                    D[n, m] = self.gamma ** (n - m)
        return D.to(self.complex_type)


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, precision="single"):
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.precision = precision
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32

        self.gammas = (1 - torch.exp(
            torch.linspace(math.log(1 / 32), math.log(1 / 512), heads, dtype=self.real_type))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.W_O = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.group_norm = ComplexGroupNorm(heads, hidden_size)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.head_size, gamma) for gamma in self.gammas
        ])

    def forward(self, X):
        if X.dtype != self.complex_type:
            X = torch.complex(X, ptu.zeros_like(X)).to(self.complex_type)

        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X[:, :, i * self.head_size:(i + 1) * self.head_size]))

        Y = torch.cat(Y, dim=2)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(X.shape)

        return (self.swish(X @ self.W_G) + Y) @ self.W_O


class RetNetEncoder(nn.Module):
    def __init__(self, context_dim, n_blocks, n_heads, is_agent=True):
        super(RetNetEncoder, self).__init__()
        self.layers = n_blocks
        self.hidden_dim = context_dim
        self.ffn_size = context_dim
        self.heads = n_heads

        self.retentions = nn.ModuleList([
            MultiScaleRetention(context_dim, n_heads)
            for _ in range(n_blocks)
        ])
        self.ffns = nn.ModuleList([
            ComplexFFN(context_dim, context_dim)
            for _ in range(n_blocks)
        ])
        self.layer_norm = ComplexLayerNorm(context_dim)
        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()
        for i in range(self.layers):
            y = self.retentions[i](self.layer_norm(x)) + x
            x = self.ffns[i](self.layer_norm(y)) + y
        o = x.mean(dim=1)
        if not self.is_agent:
            o = o.view(meta_batch, batch_size, context_dim)
        return o.real.float()

