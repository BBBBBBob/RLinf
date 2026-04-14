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

import os

import jax
import numpy as np
import torch
from omegaconf import DictConfig

import rlinf.algorithms  # noqa: F401
import openpi.training.data_loader as openpi_data_loader
from rlinf.algorithms.registry import policy_loss
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

class FSDPDiscWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.data_loader, self.negative_data_loader, self.data_config = self.build_dataloader()
    
        self.data_iter = iter(self.data_loader)
        self.negative_data_iter = iter(self.negative_data_loader)
        self.global_step = 0

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            # self.offload_param_and_grad()
            # self.offload_optimizer()
            self.offload_param_and_grad()
            self.offload_discriminator_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            if self.cfg.runner.get("ckpt_path", None):
                model_dict = torch.load(self.cfg.runner.ckpt_path)
                model.load_state_dict(model_dict)
            return model
        return super().model_provider_func()

    def build_dataloader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI, SupportedModel.IRLOPENPI]:
            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name + "_positive",
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )

            negative_config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name + "_negative",
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            )

            negative_data_loader = openpi_data_loader.create_data_loader(
                negative_config, framework="pytorch", shuffle=True
            )

            return data_loader, negative_data_loader, {
                "positive": config,
                "negative": negative_config,
            }

        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def run_training(self):
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_discriminator_optimizer(self.device)
            self.model.freeze_vlm()
            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
          
            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            metrics = {}

            avg_loss = 0.0
            irl_loss_type = self.cfg.algorithm.get("irl_loss_type", "gail")
            irl_entropy_bonus = self.cfg.algorithm.get("irl_entropy_bonus", 0.0)
            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                observation, actions = next(self.data_iter)
                negative_observation, negative_actions = next(self.negative_data_iter)

                observation = jax.tree.map(
                    lambda x: torch.as_tensor(x, device=self.device)
                    .contiguous()
                    .clone(),
                    observation,
                )
                actions = actions.to(torch.float32)
                actions = actions.to(self.device)

                negative_observation = jax.tree.map(
                    lambda x: torch.as_tensor(x, device=self.device)
                    .contiguous()
                    .clone(),
                    negative_observation,
                )
                negative_actions = negative_actions.to(torch.float32)
                negative_actions = negative_actions.to(self.device)

                with self.amp_context:
                    positive_batch = {
                        "observation": observation,
                        "normalized_actions": actions.squeeze(1),
                    }
                    negative_batch = {
                        "observation": negative_observation,
                        "normalized_actions": negative_actions.squeeze(1),
                    }
                    positive_disc_output = self.model(
                        data=positive_batch,
                        head_name="discriminator",
                        data_type="expert",
                    )
                    negative_disc_output = self.model(
                        data=negative_batch,
                        head_name="discriminator",
                        data_type="expert",
                    )
                    positive_target = torch.ones_like(positive_disc_output)
                    negative_target = torch.zeros_like(negative_disc_output)
                    loss_kwargs = {
                        "loss_type": irl_loss_type,
                        "task_type": self.cfg.runner.task_type,
                        "negative_input": negative_disc_output,
                        "negative_target": negative_target,
                        "positive_input": positive_disc_output,
                        "positive_target": positive_target,
                    }
                    loss, disc_metrics = policy_loss(**loss_kwargs)

                    if irl_entropy_bonus > 0:
                        entropy_loss, entropy_metrics = policy_loss(
                            loss_type=f"{irl_loss_type}_entropy",
                            task_type=self.cfg.runner.task_type,
                            negative_input=negative_disc_output,
                            positive_input=positive_disc_output,
                        )
                        disc_metrics.update(entropy_metrics)
                        loss -= irl_entropy_bonus * entropy_loss

                total_loss = loss.detach().item()
                loss = loss / self.gradient_accumulation
                avg_loss += loss.item()
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()
                disc_metrics["discriminator/total_loss"] = total_loss
                append_to_dict(metrics, disc_metrics)

            grad_norm, lr_list = self.discriminator_optimizer_step()
            self.discriminator_optimizer.zero_grad(set_to_none=True)

            # Collect stats
            lr_value = (
                lr_list[0]
                if len(lr_list) > 0
                else self.discriminator_optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "loss": avg_loss,
                    "discriminator/lr": lr_value,
                    "discriminator/grad_norm": grad_norm_value,
                },
            )

            self.discriminator_lr_scheduler.step()

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
