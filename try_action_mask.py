import argparse
import os
import pathlib

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import warnings
from conf import HappyCodeConfig
from happycode.dataset import make_sft_data_modlue
from happycode.model import find_all_linear_names_of_llm
from happycode.model.deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from happycode.utils import get_logger, rank0_log, safe_save_model_for_hf_trainer, seed_everything


local_rank = 0
# with open("dict_action.pkl", "rb") as f1:
#     dic = pickle.load(f1)


# special_tokens_list = []
# for key, value in dic.items():
#     special_tokens_list.append(value)
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                warnings.warn(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def main(cfg: HappyCodeConfig) -> None:
    global local_rank
    logger = get_logger(__name__, cfg.log)
    seed_everything(cfg.training.seed)

    rank0_log(local_rank, logger, OmegaConf.to_yaml(cfg))

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(cfg.model.model_path)  # type: ignore
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<action>"]})
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [str(i) for i in range(8641)]})
    # processor.tokenizer.add_special_tokens(
    #     {"additional_special_tokens": ["<a>", "</a>", "<action>", "<x>", "</x>", "<y>", "</y>"]}
    # )
    # processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        attn_implementation=None if cfg.model.attn_implementation == "none" else cfg.model.attn_implementation,
    )
    model.language_model.resize_token_embeddings(len(processor.tokenizer))

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

    training_args = TrainingArguments(
        run_name=cfg.run_name,
        output_dir=f"{cfg.ckpt_dir}/{cfg.run_name}",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        **dict(cfg.training),
    )

    training_args.local_rank = local_rank
    model.vision_model = model.vision_model.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    model.aligner = model.aligner.to(device=training_args.device)

    # # data module
    data_module = make_sft_data_modlue(processor, cfg.dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=processor.tokenizer,
        **data_module,
    )

    ckpt_dir = f"{cfg.ckpt_dir}/{cfg.run_name}"
    if list(pathlib.Path(ckpt_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if lora_cfg.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_cfg.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.model_config.save_pretrained(training_args.output_dir)
            processor.tokenizer.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
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
