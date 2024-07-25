import json
import os
from dataclasses import dataclass
from typing import Any

import PIL
from torch.utils.data import Dataset

from conf.default import BaseDatasetConfig
from happycode.model.deepseek_vl.utils.io import load_pil_images


def load_pil_images_from_path(image_list: list[str]) -> list[PIL.Image.Image]:
    """
    Support file path or base64 images.

    Args:
        image_list (List[str]): the list of image paths.
    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.
    """

    return [PIL.Image.open(image_data).convert("RGB") for image_data in image_list]


class ActionSftDataset(Dataset):
    def __init__(self, vl_chat_processor, dataset_cfg: BaseDatasetConfig) -> None:
        super(__class__, self).__init__()

        self.chat_processor = vl_chat_processor
        self.dataset_cfg = dataset_cfg

        self.sft_file = os.path.join(dataset_cfg.data_dir, dataset_cfg.file)

        self.sft_data = json.load(open(self.sft_file))[:20]

    def __len__(self):
        return len(self.sft_data)

    def __getitem__(self, index) -> Any:
        """
        return format:
        {
            "sft_format": "prompt",
            "input_ids": tensor([100000,....]),
            "labels": tensor([-100,-100, xxx])
            "pixel_values": tensor([1, 3, h, w]),
            "num_image_tokens": tensor([576])
        }
        """
        data = self.sft_data[index]["conversations"]
        history = data[0]["history"]

        pil_images = load_pil_images(data)

        prepare = self.chat_processor(conversations=data, images=pil_images, force_batchify=False)
        # todo: history["actions"] -> input_ids(list)
        history_action_input_ids = [self.chat_processor.tokenizer.encode(action) for action in history["actions"]]

        prepare["history"] = {
            "images": self.chat_processor.image_processor(
                load_pil_images_from_path(history["images"]), return_tensors="pt"
            ).pixel_values,
            "actions": history_action_input_ids,
        }
        return prepare


@dataclass
class SFTDataCollator:
    vl_chat_processor: Any

    def __call__(self, batch):
        r = self.vl_chat_processor.batchify(batch)
        return r


def make_action_sft_data_modlue(vl_chat_processor, dataset_cfg: BaseDatasetConfig) -> dict[str, Any]:
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = ActionSftDataset(vl_chat_processor, dataset_cfg)
    data_collator = SFTDataCollator(vl_chat_processor)

    return {
        "train_dataset": sft_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
