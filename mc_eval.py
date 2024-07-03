import hydra
from transformers import (
    AutoModelForCausalLM,
)

from conf import HappyCodeConfig
from env import custom_env_register, make_custom_env
from model import MultiModalityCausalLM, VLChatProcessor


@hydra.main(version_base=None, config_name="happy", config_path="conf")
def main(cfg: HappyCodeConfig):
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(cfg.model.model_path)  # type: ignore

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        attn_implementation=None if cfg.model.attn_implementation == "none" else cfg.model.attn_implementation,
    )
    model.eval()
    print("Load Model")

    env_config = cfg.env
    custom_env_register(
        env_config.name, env_config.max_episode_steps, env_config.preferred_spawn_biome, env_config.inventory
    )

    env = make_custom_env(env_name=env_config.name)

    task = "chop a tree"
    done = False
    obs = env.reset()
    while not done:
        action = model.generate()
        obs, reward, done, info = env.step(action)

    env.save("xxx.mp4")

    print("custom env ok")


if __name__ == "__main__":
    main()
