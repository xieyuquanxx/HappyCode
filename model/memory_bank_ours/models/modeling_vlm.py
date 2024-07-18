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


import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.models.blip_2 import Blip2QFormerConfig

from .clip_encoder import CLIPVisionTower, HybridVisionTower
from .projector import MlpProjector


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
    model_type = "mb_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "mb_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MemoryBankQformerConfig(Blip2QFormerConfig):
    memory_bank_length: int = 100
    num_frames: int = 100
    num_query_tokens: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.memory_bank_length = kwargs.get("memory_bank_length", 100)
        self.num_frames = kwargs.get("num_frames", 100)
        self.num_query_tokens = kwargs.get("num_query_tokens", 32)


class MultiModalityConfig(PretrainedConfig):
    model_type = "mb_multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    language_config: LlamaConfig
    qformer_config: MemoryBankQformerConfig

    cofig: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        language_config = kwargs.get("language_config", {})
        # if isinstance(language_config, LlamaConfig):
        #     self.language_config = language_config
        # else:
        self.language_config = LlamaConfig(**language_config)
        self.qformer_config = MemoryBankQformerConfig(**kwargs.get("qformer_config", {}))

        self.config = self.language_config


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "mb_multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    main_input_name: str = "inputs_embeds"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    _supports_flash_attn_2: bool = True

    def __init__(self, config: MultiModalityConfig, **kargs):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)  # type: ignore

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

        self.config = language_config
        self.model_config = config

        # qformer_config = kargs.get("qformer_config", None)
        # # todo: hyperparameter设置
        # if qformer_config is None:
        #     self.qformer_config = MemoryBankQformerConfig(
        #         encoder_hidden_size=1024,
        #         hidden_size=1024,
        #         vocab_size=language_config.vocab_size,
        #         num_attention_heads=16,
        #     )
        # else:
        #     self.qformer_config = MemoryBankQformerConfig(**qformer_config)

        #! add qformer
        # self.query_tokens = nn.Parameter(
        #     torch.zeros(1, self.qformer_config.num_query_tokens, self.qformer_config.hidden_size)
        # )
        # self.qformer = Blip2QFormerModel(self.qformer_config)
        # self.qformer.encoder = apply_memory_bank(
        #     self.qformer.encoder, self.qformer_config.memory_bank_length, self.qformer_config.num_frames
        # )
        # self.query_tokens.data.normal_(mean=0.0, std=self.qformer_config.initializer_range)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
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

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        # images_embeds = self.aligner(self.vision_model(images))
        images_features = self.vision_model(images)  # [b*n, 576, 1024]

        # [b x n, T2, D] -> [b, n x T2, D]
        # images_embeds = rearrange(images_features, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        # images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        image_attention_mask = torch.ones(
            images_features.size()[:-1], dtype=torch.long, device=images_features.device
        )

        query_tokens = self.query_tokens.expand(images_features.shape[0], -1, -1).to(
            images_features.device, torch.bfloat16
        )  # [bs*n, 32, hidden_size]
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=images_features.to(torch.bfloat16),
            encoder_attention_mask=image_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # todo: aligne是1个MLP，是否需要改成Linear
        images_embeds = self.aligner(query_output)  # [bs*n, num_query_token, 2048]
        images_embeds = rearrange(
            images_embeds, "(b n) t d -> (b n t) d", b=bs, n=n
        )  # [bs* n*num_query_token, 2048]
        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        # select 32 tokens from the image embeddings

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds[images_seq_mask] = images_embeds.to(dtype=inputs_embeds.dtype)

        return inputs_embeds

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
            inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values, images_seq_mask, images_emb_mask)
        # input_ids 和 inputs_embeds 不能同时为空，也不能同时都不为空
        return self.language_model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


AutoConfig.register("mb_vision", VisionConfig)
AutoConfig.register("mb_aligner", AlignerConfig)
AutoConfig.register("mb_multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
print("Yes")
