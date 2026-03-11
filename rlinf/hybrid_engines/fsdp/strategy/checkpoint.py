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

from typing import Iterable, Mapping, Union

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion
from rlinf.utils.utils import get_rng_state, set_rng_state


class Checkpoint(Stateful):
    def __init__(
        self,
        model: Union[FSDP, FSDPModule],
        optimizer: Union[Optimizer, Mapping[str, Optimizer]],
        lr_scheduler: Union[LRScheduler, Mapping[str, LRScheduler]],
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.opts = opts
        self.fsdp_version = fsdp_version

    def _torch_optimizers(self) -> Union[Optimizer, Iterable[Optimizer]]:
        if isinstance(self.optimizer, Mapping):
            return tuple(self.optimizer.values())
        return self.optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(
            model=self.model, optimizers=self._torch_optimizers(), options=self.opts
        )
        out = {"model": model_sd, "optim": optim_sd, "fsdp_version": self.fsdp_version.value}
        if isinstance(self.lr_scheduler, Mapping):
            out["lr_scheduler"] = {
                name: scheduler.state_dict()
                for name, scheduler in self.lr_scheduler.items()
            }
        else:
            out["lr_scheduler"] = self.lr_scheduler.state_dict()
        out["rng"] = get_rng_state()
        return out

    def load_state_dict(self, state):
        assert "fsdp_version" in state, "Checkpoint is missing FSDP version info."
        ckpt_fsdp_version = FSDPVersion(state["fsdp_version"])
        if ckpt_fsdp_version != self.fsdp_version:
            raise ValueError(
                f"FSDP version mismatch: checkpoint version {ckpt_fsdp_version} != current version {self.fsdp_version}"
            )
        optim_state = state["optim"]
        if isinstance(self.optimizer, Mapping) and not isinstance(optim_state, dict):
            if "main" in self.optimizer:
                optim_state = {"main": optim_state}
            else:
                first_key = next(iter(self.optimizer))
                optim_state = {first_key: optim_state}
        set_state_dict(
            model=self.model,
            optimizers=self._torch_optimizers(),
            model_state_dict=state["model"],
            optim_state_dict=optim_state,
            options=self.opts,
        )
        if "lr_scheduler" in state:
            if isinstance(self.lr_scheduler, Mapping):
                sched_state = state["lr_scheduler"]
                if isinstance(sched_state, dict):
                    for name, scheduler in self.lr_scheduler.items():
                        if name in sched_state:
                            scheduler.load_state_dict(sched_state[name])
                else:
                    if "main" in self.lr_scheduler:
                        self.lr_scheduler["main"].load_state_dict(sched_state)
                    else:
                        next(iter(self.lr_scheduler.values())).load_state_dict(
                            sched_state
                        )
            else:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if "rng" in state:
            set_rng_state(state["rng"])
