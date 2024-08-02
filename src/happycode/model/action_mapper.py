import abc
import itertools
from collections import OrderedDict
from typing import Dict, List
from gym3.types import DictType, Discrete, TensorType

import numpy as np
import attr
import minerl.herobraine.hero.mc as mc
import numpy as np
import inspect
import functools


def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults) :], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


class Buttons:
    ATTACK = "attack"
    BACK = "back"
    FORWARD = "forward"
    JUMP = "jump"
    LEFT = "left"
    RIGHT = "right"
    SNEAK = "sneak"
    SPRINT = "sprint"
    USE = "use"
    DROP = "drop"
    INVENTORY = "inventory"

    ALL = [
        ATTACK,
        BACK,
        FORWARD,
        JUMP,
        LEFT,
        RIGHT,
        SNEAK,
        SPRINT,
        USE,
        DROP,
        INVENTORY,
    ] + [f"hotbar.{i}" for i in range(1, 10)]


class SyntheticButtons:
    # Composite / scripted actions
    CHANNEL_ATTACK = "channel-attack"

    ALL = [CHANNEL_ATTACK]


class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"


@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy


class ActionTransformer:
    """Transforms actions between internal array and minerl env format."""

    @store_args
    def __init__(
        self,
        camera_maxval=10,
        camera_binsize=2,
        camera_quantization_scheme="linear",
        camera_mu=5,
    ):
        self.quantizer = CameraQuantizer(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            quantization_scheme=camera_quantization_scheme,
            mu=camera_mu,
        )

    def camera_zero_bin(self):
        return self.camera_maxval // self.camera_binsize

    def discretize_camera(self, xy):
        return self.quantizer.discretize(xy)

    def undiscretize_camera(self, pq):
        return self.quantizer.undiscretize(pq)

    def item_embed_id_to_name(self, item_id):
        return mc.MINERL_ITEM_MAP[item_id]

    def dict_to_numpy(self, acs):
        """
        Env format to policy output format.
        """
        act = {
            "buttons": np.stack([acs.get(k, 0) for k in Buttons.ALL], axis=-1),
            "camera": self.discretize_camera(acs["camera"]),
        }
        if not self.human_spaces:
            act.update(
                {
                    "synthetic_buttons": np.stack([acs[k] for k in SyntheticButtons.ALL], axis=-1),
                    "place": self.item_embed_name_to_id(acs["place"]),
                    "equip": self.item_embed_name_to_id(acs["equip"]),
                    "craft": self.item_embed_name_to_id(acs["craft"]),
                }
            )
        return act

    def numpy_to_dict(self, acs):
        """
        Numpy policy output to env-compatible format.
        """
        assert acs["buttons"].shape[-1] == len(
            Buttons.ALL
        ), f"Mismatched actions: {acs}; expected {len(Buttons.ALL)}:\n(  {Buttons.ALL})"
        out = {name: acs["buttons"][..., i] for (i, name) in enumerate(Buttons.ALL)}

        out["camera"] = self.undiscretize_camera(acs["camera"])

        return out

    def policy2env(self, acs):
        acs = self.numpy_to_dict(acs)
        return acs

    def env2policy(self, acs):
        nbatch = acs["camera"].shape[0]
        dummy = np.zeros((nbatch,))
        out = {
            "camera": self.discretize_camera(acs["camera"]),
            "buttons": np.stack([acs.get(k, dummy) for k in Buttons.ALL], axis=-1),
        }
        return out


class ActionMapping(abc.ABC):
    """Class that maps between the standard MC factored action space and a new one you define!

    :param n_camera_bins: Need to specify this to define the original ac space for stats code
    """

    # This is the default buttons groups, it can be changed for your action space
    BUTTONS_GROUPS = OrderedDict(
        hotbar=["none"] + [f"hotbar.{i}" for i in range(1, 10)],
        fore_back=["none", "forward", "back"],
        left_right=["none", "left", "right"],
        sprint_sneak=["none", "sprint", "sneak"],
        use=["none", "use"],
        drop=["none", "drop"],
        attack=["none", "attack"],
        jump=["none", "jump"],
    )

    def __init__(self, n_camera_bins: int = 11):
        assert n_camera_bins % 2 == 1, "n_camera_bins should be odd"
        self.n_camera_bins = n_camera_bins
        self.camera_null_bin = n_camera_bins // 2
        self.stats_ac_space = DictType(
            **{
                "buttons": TensorType(shape=(len(Buttons.ALL),), eltype=Discrete(2)),
                "camera": TensorType(shape=(2,), eltype=Discrete(n_camera_bins)),
            }
        )

    @abc.abstractmethod
    def from_factored(self, ac: Dict) -> Dict:
        """Converts a factored action (ac) to the new space

        :param ac: Dictionary of actions that must have a batch dimension
        """
        pass

    @abc.abstractmethod
    def to_factored(self, ac: Dict) -> Dict:
        """Converts an action in the new space (ac) to the factored action space.

        :param ac: Dictionary of actions that must have a batch dimension
        """
        pass

    @abc.abstractmethod
    def get_action_space_update(self):
        """Return a magym (gym3) action space. This will be used to update the env action space."""
        pass

    @abc.abstractmethod
    def get_zero_action(self):
        """Return the zero or null action for this action space"""
        pass

    def factored_buttons_to_groups(self, ac_buttons: np.ndarray, button_group: List[str]) -> List[str]:
        """For a mutually exclusive group of buttons in button_group, find which option
        in the group was chosen. Assumes that each button group has the option of 'none'
        meaning that no button in the group was pressed.

        :param ac_buttons: button actions from the factored action space. Should dims [B, len(Buttons.ALL)]
        :param button_group: List of buttons in a mutually exclusive group. Each item in the
            list should appear in Buttons.ALL except for the special case 'none' which means
            no button in the group was pressed. e.g. ['none', 'forward', 'back']. For now
            'none' must be the first element of button_group

        Returns a list of length B, where each element is an item from button_group.
        """
        assert ac_buttons.shape[1] == len(
            Buttons.ALL
        ), f"There should be {len(Buttons.ALL)} buttons in the factored buttons space"
        assert button_group[0] == "none", "This function only works if 'none' is in button_group"
        # Actions in ac_buttons with order according to button_group
        group_indices = [Buttons.ALL.index(b) for b in button_group if b != "none"]
        ac_choices = ac_buttons[:, group_indices]

        # Special cases for forward/back, left/right where mutual press means do neither
        if "forward" in button_group and "back" in button_group:
            ac_choices[np.all(ac_choices, axis=-1)] = 0
        if "left" in button_group and "right" in button_group:
            ac_choices[np.all(ac_choices, axis=-1)] = 0
        ac_non_zero = np.where(ac_choices)
        ac_choice = ["none" for _ in range(ac_buttons.shape[0])]
        # Iterate over the non-zero indices so that if two buttons in a group were pressed at the same time
        # we give priority to the button later in the group. E.g. if hotbar.1 and hotbar.2 are pressed during the same
        # timestep, hotbar.2 is marked as pressed
        for index, action in zip(ac_non_zero[0], ac_non_zero[1]):
            ac_choice[index] = button_group[action + 1]  # the zero'th index will mean no button pressed
        return ac_choice


class IDMActionMapping(ActionMapping):
    """For IDM, but essentially this is just an identity mapping"""

    def from_factored(self, ac: Dict) -> Dict:
        return ac

    def to_factored(self, ac: Dict) -> Dict:
        return ac

    def get_action_space_update(self):
        """Return a magym (gym3) action space. This will be used to update the env action space."""
        return {
            "buttons": TensorType(shape=(len(Buttons.ALL),), eltype=Discrete(2)),
            "camera": TensorType(shape=(2,), eltype=Discrete(self.n_camera_bins)),
        }

    def get_zero_action(self):
        raise NotImplementedError()


class CameraHierarchicalMapping(ActionMapping):
    """Buttons are joint as in ButtonsJointMapping, but now a camera on/off meta action is added into this joint space.
    When this meta action is triggered, the separate camera head chooses a camera action which is also now a joint space.

    :param n_camera_bins: number of camera bins in the factored space
    """

    # Add camera meta action to BUTTONS_GROUPS
    BUTTONS_GROUPS = ActionMapping.BUTTONS_GROUPS.copy()
    BUTTONS_GROUPS["camera"] = ["none", "camera"]
    BUTTONS_COMBINATIONS = list(itertools.product(*BUTTONS_GROUPS.values())) + ["inventory"]
    BUTTONS_COMBINATION_TO_IDX = {comb: i for i, comb in enumerate(BUTTONS_COMBINATIONS)}
    BUTTONS_IDX_TO_COMBINATION = {i: comb for i, comb in enumerate(BUTTONS_COMBINATIONS)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_groups = OrderedDict(
            camera_x=[f"camera_x{i}" for i in range(self.n_camera_bins)],
            camera_y=[f"camera_y{i}" for i in range(self.n_camera_bins)],
        )
        self.camera_combinations = list(itertools.product(*self.camera_groups.values()))
        self.camera_combination_to_idx = {comb: i for i, comb in enumerate(self.camera_combinations)}
        self.camera_idx_to_combination = {i: comb for i, comb in enumerate(self.camera_combinations)}
        self.camera_null_idx = self.camera_combination_to_idx[
            (f"camera_x{self.camera_null_bin}", f"camera_y{self.camera_null_bin}")
        ]
        self._null_action = {
            "buttons": self.BUTTONS_COMBINATION_TO_IDX[tuple("none" for _ in range(len(self.BUTTONS_GROUPS)))]
        }
        self._precompute_to_factored()

    def _precompute_to_factored(self):
        """Precompute the joint action -> factored action matrix."""
        button_dim = self.stats_ac_space["buttons"].size
        self.BUTTON_IDX_TO_FACTORED = np.zeros((len(self.BUTTONS_IDX_TO_COMBINATION), button_dim), dtype=int)
        self.BUTTON_IDX_TO_CAMERA_META_OFF = np.zeros((len(self.BUTTONS_IDX_TO_COMBINATION)), dtype=bool)
        self.CAMERA_IDX_TO_FACTORED = np.zeros((len(self.camera_idx_to_combination), 2), dtype=int)

        # Pre compute Buttons
        for jnt_ac, button_comb in self.BUTTONS_IDX_TO_COMBINATION.items():
            new_button_ac = np.zeros(len(Buttons.ALL), dtype="i")
            if button_comb == "inventory":
                new_button_ac[Buttons.ALL.index("inventory")] = 1
            else:
                for group_choice in button_comb[:-1]:  # Last one is camera
                    if group_choice != "none":
                        new_button_ac[Buttons.ALL.index(group_choice)] = 1

                if button_comb[-1] != "camera":  # This means camera meta action is off
                    self.BUTTON_IDX_TO_CAMERA_META_OFF[jnt_ac] = True
            self.BUTTON_IDX_TO_FACTORED[jnt_ac] = new_button_ac

        # Pre compute camera
        for jnt_ac, camera_comb in self.camera_idx_to_combination.items():
            new_camera_ac = np.ones((2), dtype="i") * self.camera_null_bin
            new_camera_ac[0] = self.camera_groups["camera_x"].index(camera_comb[0])
            new_camera_ac[1] = self.camera_groups["camera_y"].index(camera_comb[1])
            self.CAMERA_IDX_TO_FACTORED[jnt_ac] = new_camera_ac

    def from_factored(self, ac: Dict) -> Dict:
        """Converts a factored action (ac) to the new space. Assumes ac has a batch dim"""
        assert ac["camera"].ndim == 2, f"bad camera label, {ac['camera']}"
        assert ac["buttons"].ndim == 2, f"bad buttons label, {ac['buttons']}"
        # Get button choices for everything but camera
        choices_by_group = OrderedDict(
            (k, self.factored_buttons_to_groups(ac["buttons"], v))
            for k, v in self.BUTTONS_GROUPS.items()
            if k != "camera"
        )
        # Set camera "on off" action based on whether non-null camera action was given
        camera_is_null = np.all(ac["camera"] == self.camera_null_bin, axis=1)
        choices_by_group["camera"] = ["none" if is_null else "camera" for is_null in camera_is_null]

        new_button_ac = []
        new_camera_ac = []
        for i in range(ac["buttons"].shape[0]):
            # Buttons
            key = tuple([v[i] for v in choices_by_group.values()])
            if ac["buttons"][i, Buttons.ALL.index("inventory")] == 1:
                key = "inventory"
            new_button_ac.append(self.BUTTONS_COMBINATION_TO_IDX[key])

            # Camera -- inventory is also exclusive with camera
            if key == "inventory":
                key = (
                    f"camera_x{self.camera_null_bin}",
                    f"camera_y{self.camera_null_bin}",
                )
            else:
                key = (f"camera_x{ac['camera'][i][0]}", f"camera_y{ac['camera'][i][1]}")
            new_camera_ac.append(self.camera_combination_to_idx[key])

        return dict(
            buttons=np.array(new_button_ac)[:, None],
            camera=np.array(new_camera_ac)[:, None],
        )

    def to_factored(self, ac: Dict) -> Dict:
        """Converts an action in the new space (ac) to the factored action space. Assumes ac has a batch dim"""
        assert ac["camera"].shape[-1] == 1
        assert ac["buttons"].shape[-1] == 1

        new_button_ac = self.BUTTON_IDX_TO_FACTORED[np.squeeze(ac["buttons"], -1)]
        camera_off = self.BUTTON_IDX_TO_CAMERA_META_OFF[np.squeeze(ac["buttons"], -1)]
        new_camera_ac = self.CAMERA_IDX_TO_FACTORED[np.squeeze(ac["camera"], -1)]
        new_camera_ac[camera_off] = self.camera_null_bin

        return dict(buttons=new_button_ac, camera=new_camera_ac)

    def get_action_space_update(self):
        return {
            "camera": TensorType(shape=(1,), eltype=Discrete(len(self.camera_combinations))),
            "buttons": TensorType(shape=(1,), eltype=Discrete(len(self.BUTTONS_COMBINATIONS))),
        }

    def get_zero_action(self):
        return self._null_action


ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=5,
    camera_quantization_scheme="mu_law",
)


class AgentActionMapper:
    def __init__(self) -> None:
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self.action_space = self.action_mapper.get_action_space_update()
        self.action_space = DictType(**self.action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    def policy2env(self, action: str) -> dict[str, np.ndarray]:
        actions = action.strip().split(" ")
        agent_output = {"buttons": np.array([[int(actions[0])]]), "camera": np.array([[int(actions[1])]])}

        agent_output = self.action_mapper.to_factored(agent_output)
        minerl_action = self.action_transformer.policy2env(agent_output)
        return minerl_action
