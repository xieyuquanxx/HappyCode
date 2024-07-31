import json
import os
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from conf.default import BaseDatasetConfig
from happycode.model.deepseek_vl.utils.io import load_pil_images


class DeepSeekSftDataset(Dataset):
    def __init__(self, processor, cfg: BaseDatasetConfig, is_eval:bool=False) -> None:
        super(__class__, self).__init__()

        self.processor = processor
        self.cfg = cfg

        self.sft_file = os.path.join(cfg.data_dir, cfg.file if not is_eval else cfg.eval_file)

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

        return self.processor(conversations=data, images=pil_images, force_batchify=False)


@dataclass
class SFTDataCollator:
    processor: Any

    def __call__(self, batch):
        return self.processor.batchify(batch)


def make_sft_data_modlue(processor, cfg: BaseDatasetConfig) -> dict[str, Any]:
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = DeepSeekSftDataset(processor, cfg)
    data_collator = SFTDataCollator(processor)
    
    eval_dataset  = None
    if cfg.eval_file is not None:
        eval_dataset = DeepSeekSftDataset(processor, cfg, is_eval=True)

    return {
        "train_dataset": sft_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
