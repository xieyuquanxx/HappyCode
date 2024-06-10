import argparse
import pathlib

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, TrainingArguments

from dataset.deepseek_vl_sft_dataset import make_sft_data_modlue
from model import DeepSeekTrainer, MultiModalityCausalLM, VLChatProcessor
from model.callback import LoggerLogCallback
from utils import get_logger, safe_save_model_for_hf_trainer

local_rank = None


def find_all_linear_names_of_llm(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # ? needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    global local_rank
    logger = get_logger(__name__, cfg)

    logger.info(cfg)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        cfg["model"]["model_path"]
    )  # type: ignore

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_path"], trust_remote_code=True
    )

    logger.info(f"Load Model from {cfg['model']['model_path']}")

    if cfg["model"]["freeze"]["vision_model"]:
        logger.info("freeze vision model")
        for param in vl_gpt.vision_model.parameters():
            param.requires_grad = False

    if cfg["model"]["freeze"]["language_model"]:
        logger.info("freeze language model")
        for param in vl_gpt.language_model.parameters():
            param.requires_grad = False

    if cfg["model"]["freeze"]["aligner"]:
        logger.info("freeze aligner")
        for param in vl_gpt.aligner.parameters():
            param.requires_grad = False

    lora_cfg = cfg["model"]["lora"]
    if lora_cfg["lora_enable"]:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_cfg["lora_r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=find_all_linear_names_of_llm(vl_gpt.language_model),
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["lora_bias"],
            task_type="CAUSAL_LM",
        )
        logger.info("Adding LoRA Adapters...")
        vl_gpt = get_peft_model(vl_gpt, lora_config)

    training_args = TrainingArguments(
        run_name=cfg["run_name"],
        output_dir=f"{cfg['ckpt_dir']}/{cfg['run_name']}",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        **cfg["training"],
    )

    training_args.local_rank = local_rank
    vl_gpt.vision_model = vl_gpt.vision_model.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    vl_gpt.aligner = vl_gpt.aligner.to(device=training_args.device)

    # # data module
    data_module = make_sft_data_modlue(vl_chat_processor, cfg["dataset"])

    trainer = DeepSeekTrainer(
        model=vl_gpt,
        args=training_args,
        tokenizer=vl_chat_processor.tokenizer,
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
            vl_gpt.config.save_pretrained(training_args.output_dir)
            vl_gpt.save_pretrained(training_args.output_dir)
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
