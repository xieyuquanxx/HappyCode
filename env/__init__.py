import gym

from .custom_minerl import CustomEnvSpec, CustomEnvWrapper, EnvRecorderWrapper


__all__ = ["CustomEnvSpec", "CustomEnvWrapper"]





def custom_env_register(
    name: str, max_episode_steps: int, preferred_spawn_biome: str, inventory: list | None = None
):
    CustomEnvSpec(
        name=name,
        max_episode_steps=max_episode_steps,
        preferred_spawn_biome=preferred_spawn_biome,
        inventory=inventory if inventory is not None else [],
    ).register()


def make_custom_env(env_name: str = "CustomEnv-v1") -> CustomEnvWrapper:
    env = gym.make(env_name)

    env = EnvRecorderWrapper(env, "videos")
    env = CustomEnvWrapper(env)
    return env
