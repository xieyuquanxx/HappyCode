import hydra
import torch
from transformers import (
    AutoModelForCausalLM,
)

from conf import HappyCodeConfig, MineRLEnvConfig
from env import custom_env_register, make_custom_env
from model import MultiModalityCausalLM, VLChatProcessor
from model.deepseek_vl.utils.io import load_pil_images


@hydra.main(version_base=None, config_name="happy", config_path="conf")
def main(cfg: HappyCodeConfig):
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(cfg.model.model_path)  # type: ignore
    device = "cuda:7"
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        attn_implementation=None if cfg.model.attn_implementation == "none" else cfg.model.attn_implementation,
    ).to(device)
    model = torch.compile(model)  # type: ignore
    model.eval()
    print("Load Model")

    env_config: MineRLEnvConfig = cfg.env  # type: ignore
    custom_env_register(**dict(env_config))  # type: ignore

    env = make_custom_env(env_name=env_config.name)

    # task = "chop a tree"
    done = False
    obs = env.reset()
    while not done:
        conversation = processor.make_single_turn_conv("prompt", "", ["obs_path * 4"])
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(
            model.device
        )
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
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

        action = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        # todo: mapping text action to minerl action
        obs, reward, done, info = env.step(action)

    env.save("xxx.mp4")

    print("custom env ok")


if __name__ == "__main__":
    main()
