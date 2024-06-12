import argparse
import pathlib
from typing import List

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    LlamaForCausalLM,
)

from dataset.deepseek_vl_sft_dataset import make_sft_data_modlue
from model import MultiModalityCausalLM, VLChatProcessor
from model.callback import LoggerLogCallback
from utils import get_logger, rank0_log, safe_save_model_for_hf_trainer

local_rank = 0


def find_all_linear_names_of_llm(model: LlamaForCausalLM) -> List[str]:
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


def main(cfg: DictConfig) -> None:
    global local_rank
    logger = get_logger(__name__, cfg)

    rank0_log(local_rank, logger, OmegaConf.to_yaml(cfg))

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        cfg["model"]["model_path"]
    )  # type: ignore

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_path"], trust_remote_code=True
    )

    rank0_log(local_rank, logger, f"Load Model from {cfg['model']['model_path']}")

    if cfg["model"]["freeze"]["vision_model"]:
        rank0_log(local_rank, logger, "freeze vision model")
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if cfg["model"]["freeze"]["language_model"]:
        rank0_log(local_rank, logger, "freeze language model")
        for param in model.language_model.parameters():
            param.requires_grad = False

    if cfg["model"]["freeze"]["aligner"]:
        rank0_log(local_rank, logger, "freeze aligner")
        for param in model.aligner.parameters():
            param.requires_grad = False

    lora_cfg = cfg["model"]["lora"]
    if lora_cfg["lora_enable"]:
        from peft.mapping import get_peft_model
        from peft.tuners.lora import LoraConfig

        lora_config = LoraConfig(
            r=lora_cfg["lora_r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=find_all_linear_names_of_llm(model.language_model),
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["lora_bias"],
            task_type="CAUSAL_LM",
        )
        if cfg["training"]["bf16"]:
            model.language_model = model.language_model.to(torch.bfloat16)
        if cfg["training"]["fp16"]:
            model.language_model = model.language_model.to(torch.float16)
        rank0_log(
            local_rank,
            logger,
            f"Adding LoRA Adapters...\nLora Config:\n {OmegaConf.to_yaml(lora_cfg)}",
        )
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        run_name=cfg["run_name"],
        output_dir=f"{cfg['ckpt_dir']}/{cfg['run_name']}",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        **cfg["training"],
    )

    training_args.local_rank = local_rank
    model.vision_model = model.vision_model.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    model.aligner = model.aligner.to(device=training_args.device)

    # # data module
    data_module = make_sft_data_modlue(processor, cfg["dataset"])

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=processor.tokenizer,
        **data_module,
    )
    trainer.add_callback(LoggerLogCallback(logger))

    ckpt_dir = f"{cfg['ckpt_dir']}/{cfg['run_name']}"
    if list(pathlib.Path(ckpt_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if lora_cfg["lora_enable"]:
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer, ckpt_dir)


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
        default="config.yaml",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="deepspeed arguments")

    args = parser.parse_args()

    cfg = compose(config_name=args.config_name, overrides=args.overrides)
    local_rank = args.local_rank

    main(cfg)
