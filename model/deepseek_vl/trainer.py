from typing import Any, Dict

from torch import Tensor
from transformers import Trainer


class DeepSeekTrainer(Trainer):
    pass
    #!FIX: floating_point_ops error
    # def floating_point_ops(self, inputs: Dict[str, Tensor | Any]):
    #     return 0
