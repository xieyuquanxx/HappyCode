import json
import os
from dataclasses import dataclass
from typing import Any

import datasets
from torch.utils.data import Dataset

from model import BaseDatasetConfig
from model.deepseek_vl.models import VLChatProcessor
from model.deepseek_vl.utils.io import load_pil_images


class DeepSeekSftDataset(Dataset):
    def __init__(
        self, vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig
    ):
        super(DeepSeekSftDataset, self).__init__()

        self.chat_processor = vl_chat_processor
        self.dataset_cfg = dataset_cfg

        self.sft_file = os.path.join(dataset_cfg.data_dir, dataset_cfg.file)

        self.sft_data = json.load(open(self.sft_file, "r"))[:1000]

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
        # if not isinstance(data, List):
        #     data = [data]
        pil_images = load_pil_images(data)

        return self.chat_processor(
            conversations=data, images=pil_images, force_batchify=False
        )


class DeepSeekDPODataset(Dataset):
    def __init__(
        self, vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig
    ):
        super(DeepSeekDPODataset, self).__init__()

        self.chat_processor = vl_chat_processor
        self.dataset_cfg = dataset_cfg

        self.file = os.path.join(dataset_cfg.data_dir, dataset_cfg.file)

        self.data = json.load(open(self.file, "r"))[:1000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
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
        assert len(data) == 3
        prompt, chosen, rejected = data[0], data[1], data[2]
        raw_prompt = [prompt, {"role": "Assistant", "content": ""}]
        chosen_data = [prompt, chosen]
        rejected_data = [prompt, rejected]
        pil_images = load_pil_images(data)
        # images_outputs = self.chat_processor.image_processor(
        #     pil_images, return_tensors="pt"
        # )
        prompt_prepare = self.chat_processor(
            conversations=raw_prompt, images=pil_images, force_batchify=True
        )
        chosen_prepare = self.chat_processor(
            conversations=chosen_data, images=pil_images, force_batchify=True
        )
        rejected_prepare = self.chat_processor(
            conversations=rejected_data, images=pil_images, force_batchify=True
        )
        return {
            "prompt_prepare": prompt_prepare,
            "chosen_prepare": chosen_prepare,
            "rejected_prepare": rejected_prepare,
        }
        return {
            "prompt": prompt_prepare.sft_format[0],
            "prompt_input_ids": prompt_prepare.input_ids,
            "prompt_attention_mask": prompt_prepare.attention_mask,
            "chosen_input_ids": chosen_prepare.input_ids,
            "chosen_attention_mask": chosen_prepare.attention_mask,
            "chosen_labels": chosen_prepare.labels,
            "rejected_input_ids": rejected_prepare.input_ids,
            "rejected_attention_mask": rejected_prepare.attention_mask,
            "rejected_labels": rejected_prepare.labels,
            "pixel_values": prompt_prepare.pixel_values,
            "chosen_images_seq_mask": chosen_prepare.images_seq_mask,
            "chosen_images_emb_mask": chosen_prepare.images_emb_mask,
            "rejected_images_seq_mask": rejected_prepare.images_seq_mask,
            "rejected_images_emb_mask": rejected_prepare.images_emb_mask,
        }


@dataclass
class SFTDataCollator(object):
    vl_chat_processor: VLChatProcessor

    def __call__(self, batch):
        return self.vl_chat_processor.batchify(batch)


@dataclass
class DPODataCollator(object):
    vl_chat_processor: VLChatProcessor

    def __call__(self, batch):
        prompt_prepares = [sample["prompt_prepare"] for sample in batch]
        chosen_prepares = [sample["chosen_prepare"] for sample in batch]
        rejected_prepares = [sample["rejected_prepare"] for sample in batch]
        prompts = [sample.sft_format[0] for sample in prompt_prepares]
        prompt_batch = self.vl_chat_processor.batchify(prompt_prepares)
        chosen_batch = self.vl_chat_processor.batchify(chosen_prepares)
        rejected_batch = self.vl_chat_processor.batchify(rejected_prepares)

        return {
            # =====no reponse=====
            "prompt": prompts,
            "prompt_input_ids": prompt_batch.input_ids,
            "prompt_attention_mask": prompt_batch.attention_mask,
            # ====chosen response=====
            "chosen_input_ids": chosen_batch.input_ids,
            "chosen_attention_mask": chosen_batch.attention_mask,
            "chosen_labels": chosen_batch.labels,
            # ====rejected response=====
            "rejected_input_ids": rejected_batch.input_ids,
            "rejected_attention_mask": rejected_batch.attention_mask,
            "rejected_labels": rejected_batch.labels,
            # ===== image ====
            "pixel_values": prompt_prepare.pixel_values,
            "chosen_images_seq_mask": chosen_batch.images_seq_mask,
            "chosen_images_emb_mask": chosen_batch.images_emb_mask,
            "rejected_images_seq_mask": rejected_batch.images_seq_mask,
            "rejected_images_emb_mask": rejected_batch.images_emb_mask,
        }


def make_sft_data_modlue(
    vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig
):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = DeepSeekSftDataset(vl_chat_processor, dataset_cfg)
    data_collator = SFTDataCollator(vl_chat_processor)

    return {
        "train_dataset": sft_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }


def make_dpo_data_modlue(
    vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig
):
    """Make dataset and collator for dpo."""
    dataset = DeepSeekDPODataset(vl_chat_processor, dataset_cfg)
    data_collator = DPODataCollator(vl_chat_processor)

    def gen(torch_dataset):
        for sample in torch_dataset:
            yield sample

    hf_dataset = datasets.Dataset.from_generator(
        gen, gen_kwargs={"torch_dataset": dataset}
    )
    return {
        "train_dataset": hf_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
