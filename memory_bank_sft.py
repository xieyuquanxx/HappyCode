import argparse
import pathlib
import pickle

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel

from conf import HappyCodeConfig
from dataset import make_sft_data_modlue
from model.callback import LoggerLogCallback
from model.memory_bank_ours.models import (
    MemoryBankQformerConfig,
    MultiModalityCausalLM,
    VLChatProcessor,
    apply_memory_bank,
)
from utils import get_logger, rank0_log, safe_save_model_for_hf_trainer, seed_everything


local_rank = 0


def find_all_linear_names_of_llm(model: LlamaForCausalLM) -> list[str]:
    """
    gate_proj, up_proj, down_proj don't need to be trained in LoRA Fine-tuning
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            if "gate" in names[-1] or "up" in names[-1] or "down" in names[-1]:
                continue
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # ? needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def main(cfg: HappyCodeConfig) -> None:
    global local_rank
    with open("dataset/dict_action.pkl", "rb") as f1:
        dic = pickle.load(f1)

    special_tokens_list = list(dic.values())

    logger = get_logger(__name__, cfg.log)
    seed_everything(cfg.training.seed)

    rank0_log(local_rank, logger, OmegaConf.to_yaml(cfg))

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(cfg.model.model_path)  # type: ignore
    processor.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<a>", "</a>", "<action>", "<x>", "</x>", "<y>", "</y>"]
            + special_tokens_list
        }
    )

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        attn_implementation=None if cfg.model.attn_implementation == "none" else cfg.model.attn_implementation,
    )

    qformer_config = MemoryBankQformerConfig(vocab_size=processor.tokenizer.vocab_size, **dict(cfg.model.qformer))

    model.model_config.qformer_config = qformer_config
    processor.num_image_tokens = qformer_config.num_query_tokens

    query_tokens = nn.Parameter(torch.zeros(1, qformer_config.num_query_tokens, qformer_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

    qformer = Blip2QFormerModel(qformer_config).to(torch.bfloat16)
    qformer.encoder = apply_memory_bank(
        qformer.encoder, qformer_config.memory_bank_length, qformer_config.num_frames
    )
    model.qformer = qformer
    model.query_tokens = query_tokens
    rank0_log(local_rank, logger, f"Load Qformer+Memory Bank with config {qformer_config}")
    rank0_log(local_rank, logger, f"Load Model from {cfg .model.model_path}")

    freeze_cfg = cfg.model.freeze
    freeze_modules_name = list(filter(lambda x: freeze_cfg[x], freeze_cfg))
    for module_name in freeze_modules_name:
        rank0_log(local_rank, logger, f"freeze {module_name}")
        model.freeze_module(module_name)

    lora_cfg = cfg.model.lora
    if lora_cfg.lora_enable:
        from peft.mapping import get_peft_model
        from peft.tuners.lora import LoraConfig

        lora_config = LoraConfig(
            r=lora_cfg.lora_r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=find_all_linear_names_of_llm(model.language_model),
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.lora_bias,  # type: ignore
            task_type="CAUSAL_LM",
        )
        if cfg.training.bf16:
            model.language_model = model.language_model.to(torch.bfloat16)  # type: ignore
        if cfg.training.fp16:
            model.language_model = model.language_model.to(torch.float16)  # type: ignore
        rank0_log(
            local_rank,
            logger,
            f"Adding LoRA Adapters...\nLora Config:\n{OmegaConf.to_yaml(lora_cfg)}",
        )
        model = get_peft_model(model, lora_config)  # type: ignore

    # if use lora, we need to set requires_grad=True for train parameters
    for param in model.qformer.parameters():
        param.requires_grad = True

    training_args = TrainingArguments(
        run_name=cfg.run_name,
        output_dir=f"{cfg.ckpt_dir}/{cfg.run_name}",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        **dict(cfg.training),
    )

    training_args.local_rank = local_rank
    model.vision_model = model.vision_model.to(training_args.device)
    model.aligner = model.aligner.to(training_args.device)
    model.qformer = model.qformer.to(training_args.device)

    # # data module
    data_module = make_sft_data_modlue(processor, cfg.dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=processor.tokenizer,
        **data_module,
    )
    rank0_log(
        local_rank, logger, f"Total parameters (M): {trainer.get_num_trainable_parameters() / 1_000_000:.2f}M"
    )
    trainer.add_callback(LoggerLogCallback(logger))

    ckpt_dir = f"{cfg.ckpt_dir}/{cfg.run_name}"
    if list(pathlib.Path(ckpt_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if lora_cfg.lora_enable:
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.model_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir)
            processor.tokenizer.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer, ckpt_dir)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.model_config.save_pretrained(training_args.output_dir)
            processor.tokenizer.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    initialize(version_base=None, config_path="conf")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "--config-name",
        "-cn",
        help="Overrides the config_name specified in hydra.main()",
        default="happy.yaml",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="deepspeed arguments")

    args = parser.parse_args()

    cfg = compose(config_name=args.config_name, overrides=args.overrides)
    local_rank = args.local_rank

    main(cfg)  # type: ignore
