from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig


class AffLlmConfig(PretrainedConfig):
    model_type = "multi_modality"
    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        # self.vision_config = VisionConfig(**vision_config)

        # aligner_config = kwargs.get("aligner_config", {})
        # self.aligner_config = AlignerConfig(**aligner_config)

        language_config = kwargs.get("language_config", {})
        # if isinstance(language_config, LlamaConfig):
        #     self.language_config = language_config
        # else:
        self.language_config = LlamaConfig(**language_config)

        # self.config = self.language_config


class AffLlmPreTrainedModel(PreTrainedModel):
    config_class = AffLlmConfig
    # base_model_prefix = "multi_modality"
    # _no_split_modules = []
    # _skip_keys_device_placement = "past_key_values"

    # main_input_name: str = "inputs_embeds"


class AffordanceVicunaOpenAD(AffLlmPreTrainedModel):
    def __init__(self, config: AffLlmConfig):
        super().__init__(config)
        language_config = config.language_config

        self.language_model = LlamaForCausalLM(language_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "???", use_fast=False, truncation_side="left"
        )
