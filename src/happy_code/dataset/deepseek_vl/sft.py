import json
import os
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from conf.default import BaseDatasetConfig
from model.deepseek_vl.utils.io import load_pil_images


class DeepSeekSftDataset(Dataset):
    def __init__(self, vl_chat_processor, dataset_cfg: BaseDatasetConfig) -> None:
        super(__class__, self).__init__()

        self.chat_processor = vl_chat_processor
        self.dataset_cfg = dataset_cfg

        self.sft_file = os.path.join(dataset_cfg.data_dir, dataset_cfg.file)

        self.sft_data = json.load(open(self.sft_file))

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

        pil_images = load_pil_images(data)

        return self.chat_processor(conversations=data, images=pil_images, force_batchify=False)


@dataclass
class SFTDataCollator:
    vl_chat_processor: Any

    def __call__(self, batch):
        return self.vl_chat_processor.batchify(batch)


def make_sft_data_modlue(vl_chat_processor, dataset_cfg: BaseDatasetConfig) -> dict[str, Any]:
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = DeepSeekSftDataset(vl_chat_processor, dataset_cfg)
    data_collator = SFTDataCollator(vl_chat_processor)

    return {
        "train_dataset": sft_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
