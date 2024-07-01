import itertools
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Self

from torch.utils.data import Dataset

from model import BaseDatasetConfig
from model.deepseek_vl.models import VLChatProcessor
from model.deepseek_vl.utils.io import load_pil_images


"""
#TODO:
1. 1个正样本+3个负样本
2. 3个正样本+3个负样本
"""


class DeepSeekDPODataset(Dataset):
    def __init__(self, vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig) -> None:
        super(DeepSeekDPODataset, self).__init__()

        self.chat_processor = vl_chat_processor
        self.dataset_cfg = dataset_cfg

        self.file = os.path.join(dataset_cfg.data_dir, dataset_cfg.file)

        self.data = json.load(open(self.file, "r"))[:1000]

    def map(self, *args, **kargs) -> Self:
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        return format:
        {
            prompt_input_ids, prompt_attention_mask
            chosen_input_ids, chosen_attention_mask, chosen_labels
            rejected_input_ids, rejected_attention_mask, rejected_labels
            pixel_values, images_seq_mask, images_emb_mask
        }
        """
        data = self.data[index]["conversations"]
        # assert len(data) == 9  # systerm + 输入输出各三个负样本 + 正样本输入输出
        # system, input_chosen, output_chosen = data[0], data[1], data[5]
        # input_rejected_1, input_rejected_2, input_rejected_3 = data[2], data[3], data[4]
        # output_rejected_1, output_rejected_2, output_rejected_3 = data[6], data[7], data[8]
        # prompt, chosen, rejected = data[0], data[1], data[2]

        system, input_chosen, output_chosen = data[0], data[1], data[3]
        rejected = list(filter(lambda x: x["role"] == "Assistant" and x["type"] == "rejected", data))
        prompt = input_chosen

        raw_prompt = [input_chosen, {"role": "Assistant", "content": ""}]
        chosen_data = [input_chosen, output_chosen]
        rejected_data = [[prompt, reject] for reject in rejected]
        pil_images = load_pil_images(data)

        self.chat_processor.system_prompt = system["content"]
        prompt_prepare = self.chat_processor(conversations=raw_prompt, images=pil_images, force_batchify=False)
        chosen_prepare = self.chat_processor(conversations=chosen_data, images=pil_images, force_batchify=False)
        rejected_prepare = [
            self.chat_processor(conversations=rejected_dt, images=pil_images, force_batchify=False)
            for rejected_dt in rejected_data
        ]
        return {
            "prompt_prepare": prompt_prepare,
            "chosen_prepare": chosen_prepare,
            "rejected_prepare": rejected_prepare,
        }


@dataclass
class DPODataCollator(object):
    vl_chat_processor: VLChatProcessor

    def __call__(self, batch) -> Dict[str, Any]:
        prompt_prepares = [sample["prompt_prepare"] for sample in batch]  # B*[]
        chosen_prepares = [sample["chosen_prepare"] for sample in batch]  # B*[prepare]
        # B*[3*[prepare]]
        rejected_prepares = [sample["rejected_prepare"] for sample in batch]
        # rejected_neg_num = len(rejected_prepares[0])
        # bs = len(batch)
        # prompts = [sample.sft_format for sample in prompt_prepares]

        prompt_batch = self.vl_chat_processor.batchify(prompt_prepares)  # B
        chosen_batch = self.vl_chat_processor.batchify(chosen_prepares)  # B
        # flatten List[List]
        rejected_prepares = list(itertools.chain(*rejected_prepares))
        rejected_batch = self.vl_chat_processor.batchify(rejected_prepares)  # 3B
        # pixel_size = prompt_batch.pixel_values.shape[1:]
        return {
            # =====no reponse=====
            # "prompt": prompts,
            # "prompt_input_ids": prompt_batch.input_ids,
            # "prompt_attention_mask": prompt_batch.attention_mask,
            # ====chosen response=====
            "chosen_input_ids": chosen_batch.input_ids,
            "chosen_attention_mask": chosen_batch.attention_mask,
            "chosen_labels": chosen_batch.labels,
            # ====rejected response=====
            "rejected_input_ids": rejected_batch.input_ids,
            "rejected_attention_mask": rejected_batch.attention_mask,
            "rejected_labels": rejected_batch.labels,
            # ===== image ====
            "pixel_values": prompt_batch.pixel_values,
            "chosen_images_seq_mask": chosen_batch.images_seq_mask,
            "chosen_images_emb_mask": chosen_batch.images_emb_mask,
            "rejected_images_seq_mask": rejected_batch.images_seq_mask,
            "rejected_images_emb_mask": rejected_batch.images_emb_mask,
        }


def make_dpo_data_modlue(vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig) -> Dict[str, Any]:
    """Make dataset and collator for dpo."""
    dataset = DeepSeekDPODataset(vl_chat_processor, dataset_cfg)
    data_collator = DPODataCollator(vl_chat_processor)

    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
