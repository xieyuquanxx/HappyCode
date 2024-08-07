from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class LoraConfig:
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )


@dataclass
class MultiModalityCausalLMBackboneConfig:
    vision_model: bool = True
    language_model: bool = True
    aligner: bool = True
    qformer: bool = True


@dataclass
class QformerConfig:
    encoder_hidden_size: int = 1024
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_query_tokens: int = 32
    memory_bank_length: int = 100
    num_frames: int = 100


@dataclass
class BaseModelConfig:
    model_path: str = "model_repo/deepseek-vl-7b-chat"
    freeze: MultiModalityCausalLMBackboneConfig = field(default_factory=MultiModalityCausalLMBackboneConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    qformer: QformerConfig = field(default_factory=QformerConfig)
    attn_implementation: str = field(
        default="none",
        metadata={"help": "Attention implementation. Can be 'none', 'flash_attention_2'"},
    )

    system_prompt: str | None = field(default=None, metadata={"help": "System prompt for generation"})


@dataclass
class BaseTrainingConfig:
    seed: int = 42
    deepspeed: str = "scripts/deepspeed/zero3.json"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1

    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = False

    fp16: bool = False
    bf16: bool = True

    num_train_epochs: int = 2
    warmup_ratio: float = 0.1
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"

    eval_strategy: str = "no"  # ['no', 'steps', 'epoch']

    save_strategy: str = "epoch"  # ['no', 'steps', 'epoch']
    save_steps: int = 5
    save_total_limit: int = 5

    log_level: str = "info"
    logging_strategy: str = "steps"
    logging_steps: int = 1

    report_to: str = "wandb"


@dataclass
class BaseDatasetConfig:
    data_dir: str = field(
        default="data",
        metadata={"help": "data directory"},
    )
    file: str = field(
        default="train.json",
        metadata={"help": "data file"},
    )
    repo: str = field(
        default="pcuenq/oxford-pets",
        metadata={"help": "hugging face repo"},
    )
    cache_dir: str = field(
        default="data/cache",
        metadata={"help": "cache directory"},
    )


@dataclass
class LogConfig:
    dir: str = "logs"
    file: str = "test.log"


@dataclass
class BaseEnvConfig:
    name: str = "MineRL-v0"


@dataclass
class HappyCodeConfig:
    project: str = MISSING
    run_name: str = "HappyCode"
    ckpt_dir: str = "ckpt"

    log: LogConfig = field(default_factory=LogConfig)

    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    training: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)  # type: ignore
    dataset: BaseDatasetConfig = field(default_factory=BaseDatasetConfig)
    env: BaseEnvConfig = field(default_factory=BaseEnvConfig)


@dataclass
class MineRLEnvConfig(BaseEnvConfig):
    max_episode_steps: int = 1200
    preferred_spawn_biome: str = "plains"
    inventory: list = field(default_factory=list)


cs = ConfigStore.instance()

cs.store(name="config", node=HappyCodeConfig)
