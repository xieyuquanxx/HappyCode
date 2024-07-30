import torch
from .action_dpo import ActionMultiModalityCausalLM
from .deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from .memory_bank.models import MemoryBankMultiModalityCausalLM

from transformers import LlamaForCausalLM


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


__all__ = [
    "MultiModalityCausalLM",
    "MemoryBankMultiModalityCausalLM",
    "VLChatProcessor",
    "ActionMultiModalityCausalLM",
]
