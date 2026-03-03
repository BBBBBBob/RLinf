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

import copy
import gc
import torch
from omegaconf import DictConfig, open_dict

from rlinf.data.embodied_io_struct import RewardOutput
from rlinf.config import SupportedModel
from rlinf.models import get_model
from rlinf.scheduler import Channel, Worker, Cluster
from rlinf.utils.placement import HybridComponentPlacement

class IRLRewardWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        reward_world_size = self.placement.get_world_size("reward")
        self.reward_weight_src_rank = self._rank % reward_world_size

    def init_worker(self):
        reward_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(reward_model_config):
            reward_model_config.precision = self.cfg.reward.model.precision
            reward_model_config.path = self.cfg.reward.model.model_path

        self.hf_model = get_model(reward_model_config)
        
        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    def get_dones_and_rewards(
        self,
        env_output: dict[str, torch.Tensor],
        pred_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards) tensors.
        """
        # First step: no rewards yet, only dones
        if env_output["rewards"] is None:
            return env_output["dones"].bool().cpu().contiguous(), None

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()
        if pred_rewards is not None:
            rewards += pred_rewards.cpu().contiguous()
        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        return dones, rewards
    

    def _predict_rewards(self, env_output: dict[str, torch.Tensor]) -> torch.Tensor:
        predict_fn = getattr(self.hf_model, "predict_reward_batch", None)
        if predict_fn is None:
            raise AttributeError("Reward prediction method is not available.")
        with torch.no_grad():
            return predict_fn(env_output)
        
    async def predict_rewards(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.rollout_epoch):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)
                    if (
                        "normalized_actions" in env_output
                        # "next_obs" in env_output
                        # and "normalized_actions" in env_output
                    ):
                        assert env_output['rewards'] is not None, "Rewards must be in the env_output"
                        ### Maybe add extracted_obs
                        pred_rewards = self._predict_rewards(env_output)
                        dones, rewards = self.get_dones_and_rewards(env_output, pred_rewards)
                        reward_output = RewardOutput(rewards=rewards, dones=dones)
                    else:
                        reward_output = RewardOutput(rewards=None, dones=env_output['dones'].bool().cpu().contiguous())
                    self.send_reward(output_channel, reward_output.to_dict())

            for _ in range(self.num_pipeline_stages):
                assert "normalized_actions" in env_output and env_output['rewards'] is not None, "env_output structure is not correct"
                env_output =  await self.recv_env_output(input_channel)
                pred_rewards = self._predict_rewards(env_output)
                dones, rewards = self.get_dones_and_rewards(env_output, pred_rewards)
                reward_output = RewardOutput(rewards=rewards, dones=dones)
                self.send_reward(output_channel, reward_output.to_dict())

    def send_reward(self, output_channel: Channel, rewards: dict[str, torch.Tensor], mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=rewards,
            key=f"{self._rank}_{mode}", async_op=True
        )

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}_reward", async_op=True
        ).async_wait()
        return env_output

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name, src_rank=self.reward_weight_src_rank, async_op=True
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()