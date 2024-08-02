import argparse
import os

import torch
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM


def merge(model_name: str, lora_path: str, new_model_name: str, device: str = "cpu") -> None:
    if "deepseek" in model_name:
        from happycode.model.deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
    else:
        raise NotImplementedError
        # from happycode.model.memory_bank.models import MultiModalityCausalLM, VLChatProcessor

    device_arg = {"device_map": {"": device}}

    print("load base model")
    base_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, **device_arg
    )

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(lora_path)  # type: ignore
    base_model.language_model.resize_token_embeddings(len(processor.tokenizer))

    try:
        non_lora_trainables = torch.load(os.path.join(lora_path, "non_lora_trainables.bin"), map_location="cpu")
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        base_model.load_state_dict(non_lora_trainables, strict=False)
    except FileNotFoundError:
        print("load checkpoint does not has non_trainables.bin")

    print("load lora model")
    adapter = PeftModel.from_pretrained(
        base_model,
        lora_path,
        # ignore_mismatched_sizes=True,
    )
    model = adapter.merge_and_unload(progressbar=True)
    model.save_pretrained(new_model_name)
    model.model_config.save_pretrained(new_model_name)
    processor.tokenizer.save_pretrained(new_model_name)
    processor.save_pretrained(new_model_name)

    print("done :)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="model_repo/deepseek-vl-7b-chat")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/deepseek_vl_7b_sft_lora/2024-06-12-14-24",
    )
    parser.add_argument("--new_model_name", type=str, default="deepseek_vl_7b_lora")
    args = parser.parse_args()

    merge(**vars(args))
