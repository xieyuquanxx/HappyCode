import argparse

import torch
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM

from model import MultiModalityCausalLM


def merge(model_name: str, lora_path: str, new_model_name: str, device: str = "cuda:0"):
    device_arg = {"device_map": {"": device}}

    base_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, **device_arg
    )
    adapter = PeftModel.from_pretrained(
        base_model,
        lora_path,
        ignore_mismatched_sizes=True,
    )
    model = adapter.merge_and_unload(progressbar=True)
    model.save_pretrained(f"checkpoints/{new_model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="model_repo/deepseek-vl-7b-chat"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/deepseek_vl_7b_sft_lora/2024-06-12-14-24",
    )
    parser.add_argument("--new_model_name", type=str, default="deepseek_vl_7b_lora")
    args = parser.parse_args()

    merge(**vars(args))
