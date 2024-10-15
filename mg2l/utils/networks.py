import torch.nn as nn
import torch.nn.functional as F

import mg2l.utils.pytorch_utils as ptu


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu,
            output_activation=ptu.identity,
            layer_norm=True,
            out_layer_norm=False,
            use_residual=False,
            device=ptu.device,
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.out_layer_norm = out_layer_norm
        self.use_residual = use_residual

        self.fcs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = ptu.m_init(nn.Linear(in_size, next_size))
            in_size = next_size
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.layer_norms.append(ln)

        self.last_fc = ptu.m_init(nn.Linear(in_size, output_size))
        if self.out_layer_norm:
            self.last_ln = nn.LayerNorm(output_size)

        self.to(device)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x1 = fc(x)
            if self.use_residual and (x.shape[-1] == x1.shape[-1]):
                x = x + self.hidden_activation(x1)
            else:
                x = self.hidden_activation(x1)

            if self.layer_norm:
                x = self.layer_norms[i](x)

        y = self.last_fc(x)
        if self.use_residual and (x.shape[-1] == y.shape[-1]):
            y = x + self.output_activation(y)
        else:
            y = self.output_activation(y)

        if self.out_layer_norm:
            y = self.last_ln(y)

        return y
