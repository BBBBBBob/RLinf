# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from rlinf.models.embodiment.modules.mlp import MLP

class DiscriminatorHead(nn.Module):
    def __init__(
            self,
            action_enc_dim: list,
            vision_enc_dim: list,
            dec_dim: list,
            activation: str= "gelu"
    ):
        super().__init__()
        self.action_encoder = MLP(dim_list=action_enc_dim, activation_type=activation)
        self.vision_encoder = MLP(dim_list=vision_enc_dim, activation_type=activation)
        self.decoder = MLP(dim_list=dec_dim, activation_type=activation)


    def forward(self, action, vision):
        action_enc = self.action_encoder(action)
        vision_enc = self.vision_encoder(vision)
        enc = torch.concatenate((action_enc, vision_enc), dim=-1)
        x = self.decoder(enc)

        return x

class DiscriminatorHead_not_use(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(512, 128),
        output_dim: int = 1,
        activation: str = "gelu",  # 'relu' or 'gelu'
        bias_last: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))

        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())


    ### TODO check last layer initialization
    def _init_weights(self, nonlinearity="relu"):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)
