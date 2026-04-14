import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from itertools import chain

import jax
import numpy as np
import torch
import torch.nn.functional as F 
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from transformers import GemmaForCausalLM

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.discriminator_head import DiscriminatorHead
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)


class OpenPi0ForRLActionRewardPrediction(OpenPi0ForRLActionPrediction):
    """
    Pi05 model for reinforcement learning and inverse reinforcement learning.
    """

    config: OpenPi0Config

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        super().__init__(config)
        # Align discriminator dimensions with backbone config (pi0 uses 1024 width, pi05 uses 2048)
        disc_vision_dim = 2048 if "pi05_" in config.config_name else 1024
        disc_action_dim = config.action_dim
        disc_hidden_dim = disc_vision_dim
     
        self.discriminator_head = DiscriminatorHead(
            action_enc_dim=[disc_action_dim, 1024, disc_hidden_dim],
            vision_enc_dim=[disc_vision_dim, disc_hidden_dim],
            dec_dim=[
                disc_hidden_dim * 2,
                disc_hidden_dim,
                disc_hidden_dim // 2,
                1,
            ],
        )
        self.disc_action_dim = disc_action_dim
        self.disc_vision_dim = disc_vision_dim
        dtype = self.action_out_proj.weight.dtype
        device = self.action_out_proj.weight.device
        self.discriminator_head = self.discriminator_head.to(dtype=dtype, device=device)
    
    def demo_action_transform(self, demo_action: dict):
        ### make sure the input here is 32?
        # split & transform
        batch_size = demo_action['actions'].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()) if torch.is_tensor(x) else x, demo_action)
            sample = self._input_transform(sample)
            transformed_samples.append(sample)
        # recombine
        demo_action = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
  
        return demo_action
    
    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation - ensure float32 to prevent BFloat16 conversion issues
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            state = env_obs["states"]
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed_obs["observation/state"] = state
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        if "next_image" in env_obs:
            processed_obs["observation/next_image"] = env_obs["next_image"]
        return processed_obs
    
    def get_prefix_output(self, observation: _model.Observation):        
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        [prefix_output, _], _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        return prefix_output

    @torch.no_grad()
    def predict_reward_batch(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        env_obs = data["chunk_observations"]
        envs, num_steps = env_obs["main_images"].shape[:2]
        env_obs = {
            "main_images": env_obs["main_images"].flatten(0, 1),
            "wrist_images": env_obs["wrist_images"].flatten(0, 1),
            "states": env_obs["states"].flatten(0, 1),
            "task_descriptions": list(chain.from_iterable(env_obs["task_descriptions"]))
        }
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs, change the keys
        processed_obs = self.input_transform(
            to_process_obs, transpose= False
        )  # policy input obs -> model input obs, normalizing the images
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        device = observation.state.device

        prefix_output = self.get_prefix_output_from_vlm(self.get_prefix_output(observation))

        action = data["normalized_actions"].to(device=device).flatten(0, 1)
        disc_out = self.discriminator_head(action, prefix_output)

        rewards = torch.maximum(torch.zeros(disc_out.shape, device=device), disc_out) + torch.log1p(torch.exp(-torch.abs(disc_out)))
        rewards = rewards.reshape(envs, num_steps).detach().cpu()

        ### todo also pass normalized actions
        chunk_observations = {
            # "chunk_observation/state": to_process_obs["observation/state"],
            # "chunk_observation/image": to_process_obs["observation/image"],
            # "chunk_observation/wrist_image": to_process_obs["observation/wrist_image"],
            # "chunk_tokenized_prompt": processed_obs["tokenized_prompt"],
            # "chunk_tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            "normalized_actions": action,
        }
        
        chunk_observations.update(to_process_obs)
        chunk_observations.pop("prompt", None)
        return rewards, chunk_observations
        # return rewards, to_process_obs["observation/next_image"].cpu().contiguous()

    def predict_action_batch(
        self, env_obs, mode: Literal["train", "eval"] = "train", compute_values=True, return_obs=True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose= False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        normalized_actions = outputs["actions"].detach().cpu().contiguous()
         
        # normalized_actions[:, :, 7:] = torch.zeros_like(normalized_actions[:, :, 7:])
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
            "normalized_actions": normalized_actions,
        }
        return actions, result
    
    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        head_name: str = "actor_critic",
        **kwargs,
    ) -> dict[str, Any]:
        if head_name == "actor_critic":
            compute_values = kwargs.get("compute_values", False)
            chains = data["chains"]
            denoise_inds = data["denoise_inds"]
            
            # input transform
            observation = self.input_transform(data, transpose=False)
            observation = _model.Observation.from_dict(observation)
            images, img_masks, lang_tokens, lang_masks, state = (
                self._preprocess_observation(observation, train=False)
            )
            # transfer to device
            device = chains.device
            images = [img.to(device) for img in images]
            img_masks = [img_mask.to(device) for img_mask in img_masks]
            state = state.to(device)
            # get log prob
            log_probs, value_t, entropy = self.get_log_prob_value(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                chains,
                denoise_inds,
                compute_values,
            )
            log_probs = log_probs[
                :, :, : self.config.action_chunk, : self.config.action_env_dim
            ]
            entropy = entropy[
                :, :, : self.config.action_chunk, : self.config.action_env_dim
            ]
            # post process
            log_probs = log_probs.mean(dim=1)
            entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
                :, None
            ]  # [:,None] to align with loss-mask shape
            value_t = value_t.mean(dim=-1, keepdim=False)

            return {
                "logprobs": log_probs,
                "values": value_t,
                "entropy": entropy,
            }
        
        elif head_name == "discriminator":
            ## If the data comes from the policy
            data_type = kwargs.get("data_type", None)
            
            if data_type == "policy":
                observation = self.input_transform(data, transpose=False)
                observation = self.precision_processor(observation)
                observation = _model.Observation.from_dict(observation)

            elif data_type == "expert": 
                observation = data["observation"]

            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            device = observation.state.device
            action = data["normalized_actions"].to(device=device)
            with torch.no_grad():
                prefix_output = self.get_prefix_output_from_vlm(self.get_prefix_output(observation))

            disc_out = self.discriminator_head(action, prefix_output).squeeze(-1)
            
            return disc_out
        
        else:
            raise ValueError(f"Invalid head name: {head_name}")
    

    def get_prefix_output_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
      
        return prefix_out_value


class OpenPi0ForRLActionRewardPredictionGemma(OpenPi0ForRLActionPrediction):
    """
    Pi05 model for reinforcement learning and inverse reinforcement learning.
    """

    config: OpenPi0Config

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        super().__init__(config)
        expert_config = copy.deepcopy(self.paligemma_with_expert.gemma_expert.config)
        self.discriminator_head = GemmaForCausalLM(config=expert_config)
        self.discriminator_head.model.embed_tokens = None
        self.discriminator_head = self.discriminator_head.to(dtype=self.action_out_proj.weight.dtype)
        width = expert_config.hidden_size
        # Dedicated projection/MLP stack for discriminator path
        self.disc_action_in_proj = torch.nn.Linear(32, width, dtype=self.action_out_proj.weight.dtype)
        self.disc_action_out_proj = torch.nn.Linear(width, 32, dtype=self.action_out_proj.weight.dtype)
        if self.pi05:
            self.disc_time_mlp_in = torch.nn.Linear(width, width, dtype=self.action_out_proj.weight.dtype)
            self.disc_time_mlp_out = torch.nn.Linear(width, width, dtype=self.action_out_proj.weight.dtype)
        else:
            self.disc_state_proj = torch.nn.Linear(32, width, dtype=self.action_out_proj.weight.dtype)
            self.disc_action_time_mlp_in = torch.nn.Linear(2 * width, width, dtype=self.action_out_proj.weight.dtype)
            self.disc_action_time_mlp_out = torch.nn.Linear(width, width, dtype=self.action_out_proj.weight.dtype)
    
    def demo_action_transform(self, demo_action: dict):
        ### make sure the input here is 32?
        # split & transform
        batch_size = demo_action['actions'].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()) if torch.is_tensor(x) else x, demo_action)
            sample = self._input_transform(sample)
            transformed_samples.append(sample)
        # recombine
        demo_action = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
  
        return demo_action
    
    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation - ensure float32 to prevent BFloat16 conversion issues
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            state = env_obs["states"]
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed_obs["observation/state"] = state
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        if "next_image" in env_obs:
            processed_obs["observation/next_image"] = env_obs["next_image"]
        return processed_obs


    @torch.no_grad()
    def predict_reward_batch(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        # env_obs = data['chunk_observations']
        ### todo env_obs key is not correct
        env_obs = data['obs']
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs, change the keys
        processed_obs = self.input_transform(
            to_process_obs, transpose= False
        )  # policy input obs -> model input obs, normalizing the images
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        device = observation.state.device
        action = data['normalized_actions'].to(device=device)
        disc_out = self._compute_gemma_discriminator(observation, action)
        rewards = torch.maximum(torch.zeros(disc_out.shape, device=device), disc_out) + torch.log1p(torch.exp(-torch.abs(disc_out)))

        ### todo need reshape the rewards
        return rewards
        # return rewards, to_process_obs["observation/next_image"].cpu().contiguous()

    def predict_action_batch(
        self, env_obs, mode: Literal["train", "eval"] = "train", compute_values=True, return_obs=True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose= False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        normalized_actions = outputs["actions"].detach().cpu().contiguous()
        # normalized_actions[:, :, 7:] = torch.zeros_like(normalized_actions[:, :, 7:])
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
            "normalized_actions": normalized_actions,
        }
        return actions, result
    
    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        head_name: str = "actor_critic",
        **kwargs,
    ) -> dict[str, Any]:
        if head_name == "actor_critic":
            compute_values = kwargs.get("compute_values", False)
            chains = data["chains"]
            denoise_inds = data["denoise_inds"]
            
            # input transform
            observation = self.input_transform(data, transpose=False)
            observation = _model.Observation.from_dict(observation)
            images, img_masks, lang_tokens, lang_masks, state = (
                self._preprocess_observation(observation, train=False)
            )
            # transfer to device
            device = chains.device
            images = [img.to(device) for img in images]
            img_masks = [img_mask.to(device) for img_mask in img_masks]
            state = state.to(device)
            # get log prob
            log_probs, value_t, entropy = self.get_log_prob_value(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                chains,
                denoise_inds,
                compute_values,
            )
            log_probs = log_probs[
                :, :, : self.config.action_chunk, : self.config.action_env_dim
            ]
            entropy = entropy[
                :, :, : self.config.action_chunk, : self.config.action_env_dim
            ]
            # post process
            log_probs = log_probs.mean(dim=1)
            entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
                :, None
            ]  # [:,None] to align with loss-mask shape
            value_t = value_t.mean(dim=-1, keepdim=False)

            return {
                "logprobs": log_probs,
                "values": value_t,
                "entropy": entropy,
            }
        
        elif head_name == "discriminator":
            ## If the data comes from the policy
            data_type = kwargs.get("data_type", None)
            
            if data_type == "policy":
                observation = self.input_transform(data, transpose=False)
                observation = self.precision_processor(observation)
                observation = _model.Observation.from_dict(observation)

            elif data_type == "expert": 
                observation = data["observation"]

            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            device = observation.state.device
            action = data["normalized_actions"].to(device=device)
            disc_out = self._compute_gemma_discriminator(observation, action)
            return disc_out
        
        else:
            raise ValueError(f"Invalid head name: {head_name}")
    

    def get_prefix_output_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
      
        return prefix_out_value

    def _compute_gemma_discriminator(
        self, observation: _model.Observation, actions: torch.Tensor
    ) -> torch.Tensor:
        """Score actions with Gemma expert conditioned on VLM key/values."""
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        action_width = self.disc_action_in_proj.in_features
        if actions.shape[-1] > action_width:
            actions = actions[..., :action_width]
        elif actions.shape[-1] < action_width:
            pad = torch.zeros(
                *actions.shape[:-1],
                action_width - actions.shape[-1],
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = torch.cat([actions, pad], dim=-1)
        actions = actions.to(dtype=torch.float32)

        timestep = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix_disc(
            state, actions, timestep
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            actions.shape[0], suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.discriminator_head.model.config._attn_implementation = "eager"  # noqa: SLF001

        disc_output = self.discriminator_head.model.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=suffix_embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        suffix_out = disc_output.last_hidden_state
        suffix_out = suffix_out[:, -actions.shape[1] :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        pred_action = self.disc_action_out_proj(suffix_out)
        target_actions = actions.to(dtype=pred_action.dtype)
        mse = torch.mean((pred_action - target_actions) ** 2, dim=[1, 2])
        return -mse

    def embed_suffix_disc(self, state, noisy_actions, timestep):
        """Embed inputs for the discriminator's Gemma expert (independent params)."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.disc_state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(st):
                return self.disc_state_proj(st)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.disc_action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy):
            return self.disc_action_in_proj(noisy)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(act_time):
                x = self.disc_action_time_mlp_in(act_time)
                x = F.silu(x)
                return self.disc_action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            def time_mlp_func(t_emb):
                x = self.disc_time_mlp_in(t_emb)
                x = F.silu(x)
                x = self.disc_time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond


### TODO Check if new config is needed
### rewrite forward and add reward model
### reward should be updated at each step
### suffix att mask is not a casual mask
# class OpenPi0ForRLActionRewardPrediction(OpenPi0ForRLActionPrediction):
#     """
#     Pi05 model for reinforcement learning and inverse reinforcement learning.
#     """

#     config: OpenPi0Config

#     def __init__(
#         self,
#         config: OpenPi0Config,
#     ):
#         super().__init__(config)
#         # self.discriminator_head = None
#         self.discriminator_head = DiscriminatorHead(
#             input_dim=1024,
#             hidden_sizes=(512, 256, 128),
#             output_dim=1,
#             activation="relu",
#             bias_last=True,
#         )
#         self.discriminator_head = self.discriminator_head.to(
#                 dtype=self.action_out_proj.weight.dtype
#             )
    
#     def demo_action_transform(self, demo_action: dict):
#         ### make sure the input here is 32?
#         # split & transform
#         batch_size = demo_action['actions'].shape[0]
#         transformed_samples = []
#         for i in range(batch_size):
#             sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()) if torch.is_tensor(x) else x, demo_action)
#             sample = self._input_transform(sample)
#             transformed_samples.append(sample)
#         # recombine
#         demo_action = jax.tree.map(
#             lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
#             *transformed_samples,
#         )
  
#         return demo_action
    
#     def obs_processor(self, env_obs):
#         # base observation
#         processed_obs = {
#             "observation/image": env_obs["main_images"],
#             "prompt": env_obs["task_descriptions"],
#         }
#         # state observation - ensure float32 to prevent BFloat16 conversion issues
#         if "calvin" in self.config.config_name:
#             state = env_obs["states"]
#             processed_obs["observation/state_ee_pos"] = state[:, :3]
#             processed_obs["observation/state_ee_rot"] = state[:, 3:6]
#             processed_obs["observation/state_gripper"] = state[:, 6:7]
#         else:
#             state = env_obs["states"]
#             if torch.is_tensor(state):
#                 state = state.to(dtype=torch.float32)
#             processed_obs["observation/state"] = state
#         # wrist image observation
#         if env_obs["wrist_images"] is not None:
#             processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
#         # store used keys
#         # if "next_image" in env_obs:
#         #     processed_obs["observation/next_image"] = env_obs["next_image"]
#         return processed_obs
    
#     def get_prefix_pad_masks_and_kv(self, observation: _model.Observation):        
#         images, img_masks, lang_tokens, lang_masks, state = (
#             self._preprocess_observation(observation, train=False)
#         )

#         prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
#             images, img_masks, lang_tokens, lang_masks
#         )
#         prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
#         prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

#         # Compute image and language key value cache
#         prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
#         self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

#         (_, _), past_key_values = self.paligemma_with_expert.forward(
#             attention_mask=prefix_att_2d_masks_4d,
#             position_ids=prefix_position_ids,
#             past_key_values=None,
#             inputs_embeds=[prefix_embs, None],
#             use_cache=True,
#         )

#         return prefix_pad_masks, past_key_values

#     @torch.no_grad()
#     def predict_reward_batch(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
#         # env_obs = data['chunk_observations']
#         env_obs = data['obs']
#         # if "next_obs" in data:
#         #     env_obs.update({"next_image": data["next_obs"]})
#         to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs, change the keys
#         processed_obs = self.input_transform(
#             to_process_obs, transpose= False
#         )  # policy input obs -> model input obs, normalizing the images
#         processed_obs = self.precision_processor(
#             processed_obs
#         )  # obs precision processor
#         observation = _model.Observation.from_dict(processed_obs)
#         prefix_pad_masks, past_key_values = self.get_prefix_pad_masks_and_kv(observation)

#         ## suffix_out batch x 10 x 1024
#         state = observation.state
#         device = state.device
#         x_t = data['normalized_actions'].to(device=device)
#         t_input = torch.zeros((x_t.shape[0],), device=device)
        
#         suffix_out = self.get_suffix_out(
#             state,
#             prefix_pad_masks,
#             past_key_values,
#             x_t,
#             t_input,
#         ).detach()   
#         disc_out = self.discriminator_head(suffix_out[:, :self.config.action_chunk]).squeeze(-1)
#         rewards = torch.maximum(torch.zeros(disc_out.shape, device=device), disc_out) + torch.log1p(torch.exp(-torch.abs(disc_out)))

#         return rewards
#         # return rewards, to_process_obs["observation/next_image"].cpu().contiguous()

#     def predict_action_batch(
#         self, env_obs, mode: Literal["train", "eval"] = "train", compute_values=True, return_obs=True
#     ) -> tuple[np.ndarray, dict[str, Any]]:
#         to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
#         processed_obs = self.input_transform(
#             to_process_obs, transpose= False
#         )  # policy input obs -> model input obs
#         processed_obs = self.precision_processor(
#             processed_obs
#         )  # obs precision processor
#         observation = _model.Observation.from_dict(processed_obs)
#         outputs = self.sample_actions(
#             observation, mode=mode, compute_values=compute_values
#         )
#         actions = self.output_transform(
#             {"actions": outputs["actions"], "state": observation.state}
#         )["actions"].numpy()
#         forward_inputs = {
#             "chains": outputs["chains"],
#             "denoise_inds": outputs["denoise_inds"],
#             "tokenized_prompt": processed_obs["tokenized_prompt"],
#             "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
#         }
#         forward_inputs.update(to_process_obs)
#         forward_inputs.pop("prompt", None)
#         normalized_actions = outputs["actions"].detach().cpu().contiguous() 
#         normalized_actions[:, :, 7:] = torch.zeros_like(normalized_actions[:, :, 7:])
#         result = {
#             "prev_logprobs": outputs["prev_logprobs"],
#             "prev_values": outputs["prev_values"],
#             "forward_inputs": forward_inputs,
#             "normalized_actions": normalized_actions,
#         }
#         return actions, result
    
#     def default_forward(
#         self,
#         data: dict[str, torch.Tensor],
#         head_name: str = "actor_critic",
#         **kwargs,
#     ) -> dict[str, Any]:
#         if head_name == "actor_critic":
#             compute_values = kwargs.get("compute_values", False)
#             chains = data["chains"]
#             denoise_inds = data["denoise_inds"]
            
#             # input transform
#             observation = self.input_transform(data, transpose=False)
#             observation = _model.Observation.from_dict(observation)
#             images, img_masks, lang_tokens, lang_masks, state = (
#                 self._preprocess_observation(observation, train=False)
#             )
#             # transfer to device
#             device = chains.device
#             images = [img.to(device) for img in images]
#             img_masks = [img_mask.to(device) for img_mask in img_masks]
#             state = state.to(device)
#             # get log prob
#             log_probs, value_t, entropy = self.get_log_prob_value(
#                 images,
#                 img_masks,
#                 lang_tokens,
#                 lang_masks,
#                 state,
#                 chains,
#                 denoise_inds,
#                 compute_values,
#             )
#             log_probs = log_probs[
#                 :, :, : self.config.action_chunk, : self.config.action_env_dim
#             ]
#             entropy = entropy[
#                 :, :, : self.config.action_chunk, : self.config.action_env_dim
#             ]
#             # post process
#             log_probs = log_probs.mean(dim=1)
#             entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
#                 :, None
#             ]  # [:,None] to align with loss-mask shape
#             value_t = value_t.mean(dim=-1, keepdim=False)

#             return {
#                 "logprobs": log_probs,
#                 "values": value_t,
#                 "entropy": entropy,
#             }
        
#         elif head_name == "discriminator":
#             ## If the data comes from the policy
#             data_type = kwargs.get("data_type", None)
            
#             if data_type == "policy":
#                 observation = self.input_transform(data, transpose=False)
#                 observation = self.precision_processor(observation)
#                 observation = _model.Observation.from_dict(observation)

#             elif data_type == "expert": 
#                 observation = data["observation"]

#             else:
#                 raise ValueError(f"Invalid data_type: {data_type}")
#             state = observation.state
#             device = state.device
#             t_input = torch.zeros((state.shape[0],), device=device)
            
#             with torch.no_grad():
#                 prefix_pad_masks, past_key_values = self.get_prefix_pad_masks_and_kv(observation)
#                 suffix_out = self.get_suffix_out(
#                     state,
#                     prefix_pad_masks,
#                     past_key_values,
#                     data["normalized_actions"],
#                     t_input,
#                 )
        
#             disc_out = self.discriminator_head(suffix_out[:, :self.config.action_chunk]).squeeze(-1)
            
#             return disc_out
        
#         else:
#             raise ValueError(f"Invalid head name: {head_name}")
        