import argparse
import pathlib

import torch
from conf import HappyCodeConfig
from happycode.dataset import make_action_sft_data_modlue
from happycode.model import ActionMultiModalityCausalLM, VLChatProcessor
from happycode.model.action_vlm.modeling_vlm import ActionQformerConfig
from happycode.model.callback import LoggerLogCallback
from happycode.model.memory_bank.models import (
    apply_memory_bank,
)
from happycode.utils import get_logger, rank0_log, safe_save_model_for_hf_trainer, seed_everything
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


local_rank = 0


def main(cfg: HappyCodeConfig) -> None:
    global local_rank

    logger = get_logger(__name__, cfg.log)
    seed_everything(cfg.training.seed)

    rank0_log(local_rank, logger, OmegaConf.to_yaml(cfg))

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(cfg.model.model_path)  # type: ignore

    model: ActionMultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        attn_implementation=None if cfg.model.attn_implementation == "none" else cfg.model.attn_implementation,
        is_sft_stage2=True,
    )

    qformer_config = ActionQformerConfig(vocab_size=processor.tokenizer.vocab_size, **dict(cfg.model.qformer))

    model.model_config.qformer_config = qformer_config

    query_tokens = nn.Parameter(torch.zeros(1, qformer_config.num_query_tokens, qformer_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

    qformer = Blip2QFormerModel(qformer_config).to(torch.bfloat16)
    qformer.encoder = apply_memory_bank(qformer.encoder, qformer_config.memory_bank_length, qformer_config.num_frames)
    qformer_fc = nn.Linear(qformer_config.fc_input_dim, qformer_config.fc_output_dim)
    qformer_fc.weight.data.normal_(mean=0.0, std=qformer_config.initializer_range)
    qformer_fc.bias.data.zero_()

    model.qformer = qformer
    model.query_tokens = query_tokens
    model.qformer_fc = qformer_fc

    rank0_log(local_rank, logger, f"Load Qformer+Memory Bank with config {qformer_config}")
    rank0_log(local_rank, logger, f"Load Model from {cfg .model.model_path}")

    freeze_cfg = cfg.model.freeze
    freeze_modules_name = list(filter(lambda x: freeze_cfg[x], freeze_cfg))
    for module_name in freeze_modules_name:
        rank0_log(local_rank, logger, f"freeze {module_name}")
        model.freeze_module(module_name)

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
    model.qformer_fc = model.qformer_fc.to(training_args.device)
    # # data module
    data_module = make_action_sft_data_modlue(processor, cfg.dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=processor.tokenizer,
        **data_module,
    )
    rank0_log(local_rank, logger, f"Total parameters (M): {trainer.get_num_trainable_parameters() / 1_000_000:.2f}M")
    trainer.add_callback(LoggerLogCallback(logger))

    ckpt_dir = f"{cfg.ckpt_dir}/{cfg.run_name}"
    if list(pathlib.Path(ckpt_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

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
