from collections.abc import Sequence

from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS

from .wrapper import BasaltTimeoutWrapper


def _custom_gym_entrypoint(
    env_spec: "CustomBaseEnvSpec",
    fake: bool = False,
):
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = BasaltTimeoutWrapper(env)
    return env


MINUTE = 20 * 60
CUSTOM_GYM_ENTRY_POINT = "env.custom_minerl.env_spec:_custom_gym_entrypoint"


class CustomBaseEnvSpec(HumanSurvival):
    LOW_RES_SIZE = 64
    HIGH_RES_SIZE = 1024

    def __init__(
        self,
        name,
        demo_server_experiment_name,
        max_episode_steps=2400,
        inventory: Sequence[dict] = (),
        preferred_spawn_biome: str = "plains",
    ) -> None:
        self.inventory = inventory  # Used by minerl.util.docs to construct Sphinx docs.
        self.preferred_spawn_biome = preferred_spawn_biome
        self.demo_server_experiment_name = demo_server_experiment_name
        super().__init__(
            name=name,
            max_episode_steps=max_episode_steps,
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=(640, 360),
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0],
        )

    def is_from_folder(self, folder: str) -> bool:
        # Implements abstractmethod.
        return folder == self.demo_server_experiment_name

    def _entry_point(self, fake: bool) -> str:
        # Don't need to inspect `fake` argument here because it is also passed to the
        # entrypoint function.
        return CUSTOM_GYM_ENTRY_POINT

    def create_observables(self):
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS),
        ] + [
            handlers.EquippedItemObservation(
                items=ALL_ITEMS,
                mainhand=True,
                offhand=True,
                armor=True,
                _default="air",
                _other="air",
            ),
            handlers.ObservationFromLifeStats(),
            handlers.IsGuiOpen(),
            # handlers.ObserveFromFullStats("use_item"),
            # handlers.ObserveFromFullStats("drop"),
            # handlers.ObserveFromFullStats("pickup"),
            # handlers.ObserveFromFullStats("break_item"),
            # handlers.ObserveFromFullStats("craft_item"),
            # handlers.ObserveFromFullStats("mine_block"),
            # handlers.ObserveFromFullStats("damage_dealt"),
            # handlers.ObserveFromFullStats("entity_killed_by"),
            # handlers.ObserveFromFullStats("kill_entity"),
            # handlers.ObserveFromFullStats(None),
        ]

    def create_agent_start(self) -> list[handlers.Handler]:
        return super().create_agent_start() + [
            handlers.PreferredSpawnBiome(self.preferred_spawn_biome),
            handlers.DoneOnDeath(),
        ]

    def create_actionables(self):
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        return super().create_actionables()

    def create_agent_handlers(self) -> list[handlers.Handler]:
        return []

    def create_server_world_generators(self) -> list[handlers.Handler]:
        # TODO the original biome forced is not implemented yet. Use this for now.
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> list[handlers.Handler]:
        return [
            # handlers.ServerQuitFromTimeUp((self.max_episode_steps * mc.MS_PER_STEP)),  # type: ignore
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def create_server_decorators(self) -> list[handlers.Handler]:
        return []

    def create_server_initial_conditions(self) -> list[handlers.Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def get_blacklist_reason(self, npz_data: dict) -> str | None:
        """
        Some saved demonstrations are bogus -- they only contain lobby frames.

        We can automatically skip these by checking for whether any snowballs
        were thrown.
        """
        # TODO(shwang): Waterfall demos should also check for water_bucket use.
        #               AnimalPen demos should also check for fencepost or fence gate use.
        # TODO Clean up snowball stuff (not used anymore)
        equip = npz_data.get("observation$equipped_items$mainhand$type")
        use = npz_data.get("action$use")
        if equip is None:
            return f"Missing equip observation. Available keys: {list(npz_data.keys())}"
        if use is None:
            return f"Missing use action. Available keys: {list(npz_data.keys())}"

        assert len(equip) == len(use) + 1, (len(equip), len(use))

        for i in range(len(use)):
            if use[i] == 1 and equip[i] == "snowball":
                return None
        return "BasaltEnv never threw a snowball"

    def create_mission_handlers(self):
        # Implements abstractmethod
        return ()

    def create_monitors(self):
        # Implements abstractmethod
        return ()

    def create_rewardables(self):
        # Implements abstractmethod
        return ()

    def determine_success_from_rewards(self, rewards: list) -> bool:
        """Implements abstractmethod.

        Basalt environment have no rewards, so this is always False."""
        return False

    def get_docstring(self):
        return self.__class__.__doc__


class CustomEnvSpec(CustomBaseEnvSpec):
    def __init__(
        self,
        name: str,
        max_episode_steps: int,
        preferred_spawn_biome: str,
        inventory: Sequence[dict] | None = None,
    ):
        super().__init__(
            name=name,
            demo_server_experiment_name=name,
            max_episode_steps=max_episode_steps,
            preferred_spawn_biome=preferred_spawn_biome,
            inventory=inventory if inventory is not None else (),
        )
