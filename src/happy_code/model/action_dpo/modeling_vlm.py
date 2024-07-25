# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from typing import Any

import torch
from attrdict import AttrDict
from einops import rearrange
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.models.blip_2 import Blip2QFormerConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel

from model.deepseek_vl.models.clip_encoder import CLIPVisionTower, HybridVisionTower
from model.deepseek_vl.models.projector import MlpProjector
from model.memory_bank.models.qformer import apply_memory_bank


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "HybridVisionTower" in cls_name:
        cls = HybridVisionTower

    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "action_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "action_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class ActionQformerConfig(Blip2QFormerConfig):
    memory_bank_length: int = 100
    num_frames: int = 100
    num_query_tokens: int = 32

    fc_input_dim: int = 1024
    fc_output_dim: int = 2048

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.memory_bank_length = kwargs.get("memory_bank_length", 100)
        self.num_frames = kwargs.get("num_frames", 100)
        self.num_query_tokens = kwargs.get("num_query_tokens", 32)
        self.fc_input_dim = kwargs.get("fc_input_dim", 1024)
        self.fc_output_dim = kwargs.get("fc_output_dim", 2048)


class ActionMultiModalityConfig(PretrainedConfig):
    model_type = "action_multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    language_config: LlamaConfig
    qformer_config: ActionQformerConfig

    cofig: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        language_config = kwargs.get("language_config", {})

        self.language_config = LlamaConfig(**language_config)
        self.qformer_config = ActionQformerConfig(**kwargs.get("qformer_config", {}))

        self.is_sft_stage1 = kwargs.get("is_sft_stage1", False)
        self.is_sft_stage2 = kwargs.get("is_sft_stage2", False)
        self.is_dpo = kwargs.get("is_dpo", False)


class ActionMultiModalityPreTrainedModel(PreTrainedModel):
    config_class = ActionMultiModalityConfig
    base_model_prefix = "action_multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    main_input_name: str = "input_ids"


class ActionMultiModalityCausalLM(ActionMultiModalityPreTrainedModel):
    _supports_flash_attn_2: bool = True
    _module_names: list[str] = ["vision_model", "aligner", "language_model", "qformer", "qformer_fc"]
    model_config: ActionMultiModalityConfig

    def __init__(self, config: ActionMultiModalityConfig):
        super().__init__(config)
        self.model_config = config
        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)  # type: ignore

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

        self.config = language_config

        if not config.is_sft_stage1 and not config.is_sft_stage2:
            self.query_tokens = nn.Parameter(
                torch.zeros(1, config.qformer_config.num_query_tokens, config.qformer_config.hidden_size)
            )
            self.query_tokens.data.normal_(mean=0.0, std=config.qformer_config.initializer_range)

            self.qformer = Blip2QFormerModel(config.qformer_config).to(torch.bfloat16)
            self.qformer.encoder = apply_memory_bank(
                self.qformer.encoder, config.qformer_config.memory_bank_length, config.qformer_config.num_frames
            )
            self.qformer_fc = nn.Linear(config.qformer_config.fc_input_dim, config.qformer_config.fc_output_dim)
            self.qformer_fc.weight.data.normal_(mean=0.0, std=config.qformer_config.initializer_range)
            self.qformer_fc.bias.data.zero_()

    def freeze_module(self, module_name: str) -> None:
        assert module_name in self._module_names, f"module_name {module_name} is invalid."
        for module in self._module_names:
            if module == module_name:
                module_var = getattr(self, module)
                for param in module_var.parameters():
                    param.requires_grad = False

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        history: dict | None = None,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """
        assert history is not None, "history is required."

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # actions shape: [bs, 8, max_action_len] -> [bs*8, max_action_len]
        history_images, history_actions = (history["images"], history["actions"])
        history_images = rearrange(history_images, "b n c h w -> (b n) c h w")
        history_actions = rearrange(history_actions, "b n t -> (b n) t")
        history_images_num = history_images.shape[0]

        images_features = self.vision_model(images)  # [b*n, 576, 1024]
        history_images_features = self.vision_model(
            history_images.to(images.device, images.dtype)
        )  # [b*n, 576, 1024]
        history_images_features = rearrange(history_images_features, "(b n) t d -> b n t d", b=bs, n=8)

        image_attention_mask = torch.ones(
            history_images_features.size()[:-1], dtype=torch.long, device=history_images_features.device
        )
        # [b*n, max_len/288, 2048] -> 最后一维度需要变成1024
        history_action_embeds = self.language_model.get_input_embeddings()(history_actions.to(images.device))
        history_action_embeds = history_action_embeds.reshape(*history_images_features.shape)  # [b, n, 576, 1024]

        # [bs, 32, hidden_size]
        query_tokens = self.query_tokens.expand(history_images_features.shape[0], -1, -1).to(
            history_images_features.device, history_images_features.dtype
        )
        for history_size in range(min(history_images_num, history_images_features.shape[1])):
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=[
                    history_images_features[:, history_size, ...],
                    history_action_embeds[:, history_size, ...],
                ],
                encoder_attention_mask=image_attention_mask,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
                # return_dict=return_dict,
            )
        query_output = query_outputs[0]  # [bs, 32, 1024]

        images_embeds_vision = self.aligner(images_features)  # [bs, 576, 2048]
        images_embeds2_qformer = self.qformer_fc(query_output)  # [bs, num_query_token, 2048]

        images_embeds = rearrange(images_embeds_vision, "b t d -> (b t) d", b=bs)  # [bs*576, 2048]
        # [bs*num_query_token, 2048]
        # images_qformer_embeds = rearrange(
        #     images_embeds2_qformer, "b t d -> (b t) d", b=bs
        # )
        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds[images_seq_mask] = images_embeds.to(dtype=inputs_embeds.dtype)

        inputs_embeds = torch.cat([images_embeds2_qformer, inputs_embeds], dim=1)

        labels = kwargs.get("labels", None)

        if labels is not None:
            ignore_qformer_embeds = torch.tensor(
                [[-100] * images_embeds2_qformer.shape[1]], device=images_embeds2_qformer.device
            )
            labels = torch.cat([ignore_qformer_embeds.repeat(bs, 1), labels], dim=1)

        return inputs_embeds, labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor | None = None,
        images_emb_mask: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kargs,
    ):
        if inputs_embeds is None:
            inputs_embeds, labels = self.prepare_inputs_embeds(
                input_ids,
                pixel_values,
                images_seq_mask,
                images_emb_mask,
                history=kargs.get("history", None),
                labels=labels,
            )
        return self.language_model.forward(
            attention_mask=torch.ones(labels.shape),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def floating_point_ops(
        self, input_dict: dict[str, torch.Tensor | Any], exclude_embeddings: bool = True
    ) -> int:
        return (
            6
            * input_dict[self.main_input_name].numel()
            * self.num_parameters(exclude_embeddings=exclude_embeddings)
        )


AutoConfig.register("action_vision", VisionConfig)
AutoConfig.register("action_aligner", AlignerConfig)
AutoConfig.register("action_multi_modality", ActionMultiModalityConfig)
AutoModelForCausalLM.register(ActionMultiModalityConfig, ActionMultiModalityCausalLM)
