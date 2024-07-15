import time
from typing import Any

import cv2
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
)

from model import MultiModalityCausalLM, VLChatProcessor
from model.deepseek_vl.utils.io import load_pil_images


def save_obs(array: np.ndarray, img_file_name: str):
    """
    Save an RGB image array to a file.

    Args:
        array (np.ndarray): The RGB image array to be saved.
        img_file_name (str): The name of the output image file.

    Returns:
        None
    """
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_file_name, array)


def make_conversations(task: str, images: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "role": "User",
            "content": task,
            "images": images,
        },
        {"role": "Assistant", "content": ""},
    ]


device = "cuda:0"
model_path = "model_repo/deepseek-vl-1.3b-chat"
processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)  # type: ignore

# todo: add special tokens
processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<a>", "</a>", "<action>"]})

processor.tokenizer.save_pretrained("test/")
processor.save_pretrained("test/")
model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
)
model = model.to(device)
model = torch.compile(model)
model.eval()
print("Load Model")

# env_config = MineRLEnvConfig()  # type: ignore
# custom_env_register(**asdict(env_config))  # type: ignore

# env = make_custom_env(env_name=env_config.name)

task = "chop a tree<image_placeholder><a><chat></a><a><chat></a><a><chat></a><a><attack><forward><x>-1.61</x><y>-1.61</y></a><a><attack><forward><x>-10.00</x><y>-10.00</y></a><a><attack><forward><right><x>-10.00</x><y>-10.00</y></a><a><x>1.61</x><y>-5.81</y></a><a><x>5.81</x><y>-3.22</y></a><a><x>3.22</x><y>-5.81</y></a><a><forward><x>0.62</x><y>-0.62</y></a><image_placeholder><a><forward><x>1.61</x><y>-3.22</y></a><a><forward><sprint><x>3.22</x><y>-5.81</y></a><a><forward><sprint><x>0.62</x><y>-10.00</y></a><a><forward><sprint><x>-0.62</x><y>-10.00</y></a><a><forward><sprint><x>0.00</x><y>-10.00</y></a><a><forward><sprint><x>0.62</x><y>-3.22</y></a><a><forward><sprint></a><a><forward><sprint></a><a><forward><sprint><x>-0.62</x><y>10.00</y></a><a><forward><jump><sprint><x>0.00</x><y>10.00</y></a><image_placeholder><a><forward><jump><sprint><x>-0.62</x><y>5.81</y></a><a><forward><jump><sprint></a><a><forward><jump><sprint></a><a><forward><jump><sprint><x>0.00</x><y>-0.62</y></a><a><forward><sprint><x>0.62</x><y>-0.62</y></a><a><forward><sprint><x>1.61</x><y>-3.22</y></a><a><forward><sprint><x>0.00</x><y>-5.81</y></a><a><forward><sprint><x>0.62</x><y>-10.00</y></a><a><forward><sprint><x>0.62</x><y>-5.81</y></a><a><forward><jump><sprint><x>0.00</x><y>-0.62</y></a><image_placeholder>"
done = False

# obs = env.reset()
# save_obs(obs["pov"], "test.png")
print("Go!")
conversation = make_conversations(task, ["test.png"] * 4)
pil_images = load_pil_images(conversation)
prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(device)
inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
times = []
for _ in range(50):
    # raw: 32.1606 s (cpu)
    # torch.compile: 21.0886 (cpu)
    # bf16: 3.6214s (gpu)
    # torch.compile + bf16: 3.4928s (gpu)
    # torch.compile + bf16 + flash-attn2: 3.4652s (gpu)
    start = time.perf_counter()
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    end = time.perf_counter()
    times.append(end - start)
    print(processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True))

t = np.mean(times)

print(f"Time taken: {t:0.4f} seconds")