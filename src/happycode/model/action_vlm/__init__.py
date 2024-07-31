import collections
import collections.abc


for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from .modeling_vlm import ActionMultiModalityCausalLM


__all__ = ["ActionMultiModalityCausalLM"]
