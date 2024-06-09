import argparse
import logging
import pathlib

import hydra
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from dataset.deepseek_vl_sft_dataset import make_sft_data_modlue
from model.callback.logger import LoggerLogCallback
from model.deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from utils import safe_save_model_for_hf_trainer

logger = logging.getLogger(__name__)

local_rank = None


# todo:
# * 1. lora       6/9
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    global local_rank
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

    # lora_cfg = cfg["training"]["lora"]
    # del cfg["training"]["lora"]
    # if lora_cfg["lora_enable"]:
    #     from peft import LoraConfig, get_peft_model

    #     raise NotImplementedError("Lora is not implemented yet.")

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

    trainer = Trainer(
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

    if cfg["training"]["lora"]["enable"]:
        raise NotImplementedError("Lora is not implemented yet.")
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
