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

from rlinf.data.io_struct import RewardOutput
from rlinf.config import SupportedModel
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Worker


class IRLRewardWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()
        self._obs_reward_queue_name = cfg.env.channel.queue_name_reward
        self._reward_queue_name = cfg.reward.channel.queue_name
        self.channel = self.connect_channel(cfg.rollout.channel.name)
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

    def init_worker(self):
        reward_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(reward_model_config):
            reward_model_config.precision = self.cfg.reward.model.precision
            reward_model_config.path = self.cfg.reward.model.model_path

        self.hf_model = get_model(reward_model_config)
        
        ### only support OpenPI
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            model_config, input_processor = get_vla_model_config_and_processor(
                self.cfg.actor
            )
            self.hf_model.setup_config_and_processor(
                model_config, self.cfg, input_processor
            )

        self.hf_model.eval()

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
        
    def predict_rewards(self):
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.rollout_epoch):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = self.recv_env_output()
                    if (
                        "next_obs" in env_output
                        and "normalized_actions" in env_output
                    ):
                        assert env_output['rewards'] is not None, "Rewards must be in the env_output"
                        # pred_rewards, processed_last_image = self._predict_rewards(env_output)
                        # dones, rewards = self.get_dones_and_rewards(env_output, pred_rewards)
                        # reward_output = RewardOutput(rewards=rewards, dones=dones, last_obs=processed_last_image)
                        pred_rewards = self._predict_rewards(env_output)
                        dones, rewards = self.get_dones_and_rewards(env_output, pred_rewards)
                        reward_output = RewardOutput(rewards=rewards, dones=dones)
                    else:
                        reward_output = RewardOutput(rewards=None, dones=env_output['dones'].bool().cpu().contiguous())
                    self.send_reward(reward_output.to_dict())

            for _ in range(self.num_pipeline_stages):
                assert "normalized_actions" in env_output and env_output['rewards'] is not None, "env_output structure is not correct"
                env_output = self.recv_env_output()
                pred_rewards = self._predict_rewards(env_output)
                dones, rewards = self.get_dones_and_rewards(env_output, pred_rewards)
                reward_output = RewardOutput(rewards=rewards, dones=dones)
                self.send_reward(reward_output.to_dict())

    def send_reward(self, rewards: dict[str, torch.Tensor]):
        self.channel.put(
            item=rewards,
            key=f"{self._reward_queue_name}_{self._rank}",
        )

    def recv_env_output(self):
        return self.channel.get(
            key=f"{self._obs_reward_queue_name}_{self._rank}",
        )
    
    def sync_model_from_actor(self):
        param_state_dict = self.recv(self.actor_group_name, src_rank=self._rank)
        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()