import argparse
import dataclasses
import pathlib
import pickle

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from trl import DPOConfig

from conf import HappyCodeConfig
from happycode.dataset import make_dpo_data_modlue
from happycode.model import MultiModalityCausalLM, VLChatProcessor, find_all_linear_names_of_llm
from happycode.model.callback import LoggerLogCallback
from happycode.trainer import VLDPOTrainer
from happycode.utils import get_logger, rank0_log, safe_save_model_for_hf_trainer, seed_everything


local_rank = 0
with open("/data/Users/xyq/developer/happy_code/dataset/dict_action.pkl", "rb") as f1:
    dic = pickle.load(f1)


special_tokens_list = []
for key, value in dic.items():
    special_tokens_list.append(value)



def main(cfg: HappyCodeConfig) -> None:
    global local_rank
    logger = get_logger(__name__, cfg.log)
    seed_everything(cfg.training.seed)

    rank0_log(local_rank, logger, OmegaConf.to_yaml(cfg))

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(cfg.model.model_path)  # type: ignore
    # add special tokens
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<a>", "</a>", "<action>", "<x>", "</x>", "<y>", "</y>"]}
    )
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        attn_implementation=None if cfg.model.attn_implementation == "none" else cfg.model.attn_implementation,
    )
    ref_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        attn_implementation=None if cfg.model.attn_implementation == "none" else cfg.model.attn_implementation,
    )
    ref_model.eval()

    rank0_log(local_rank, logger, f"Load Model from {cfg.model.model_path}")

    if cfg.model.freeze.vision_model:
        rank0_log(local_rank, logger, "freeze vision model")
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if cfg.model.freeze.language_model:
        rank0_log(local_rank, logger, "freeze language model")
        for param in model.language_model.parameters():
            param.requires_grad = False

    if cfg.model.freeze.aligner:
        rank0_log(local_rank, logger, "freeze aligner")
        for param in model.aligner.parameters():
            param.requires_grad = False

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

    training_args = DPOConfig(
        run_name=cfg.run_name,
        output_dir=f"{cfg.ckpt_dir}/{cfg.run_name}",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        padding_value=0,
        **dict(cfg.training),  # type: ignore
    )

    model.vision_model = model.vision_model.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    model.aligner = model.aligner.to(device=training_args.device)
    # data module
    data_module = make_dpo_data_modlue(processor, cfg.dataset)

    trainer = VLDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        tokenizer=processor.tokenizer,
        padding_value=0,
        **data_module,
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
            model.config.save_pretrained(training_args.output_dir)
            processor.tokenizer.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
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
        default="happy.yaml",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="deepspeed arguments")

    args = parser.parse_args()

    cfg = compose(config_name=args.config_name, overrides=args.overrides)
    local_rank = args.local_rank

    main(cfg)  # type: ignore
