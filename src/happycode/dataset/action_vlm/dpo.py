import itertools
import json
import os
from typing import Any, Self

from torch.utils.data import Dataset
from trl.trainer.utils import DPODataCollatorWithPadding

from happycode.config import BaseDatasetConfig
from happycode.model.deepseek_vl.models import VLChatProcessor
from happycode.model.deepseek_vl.utils.io import load_pil_images
from happycode.utils.image import load_pil_images_from_path


class ActionDPODataset(Dataset):
    def __init__(self, vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig) -> None:
        super(__class__, self).__init__()

        self.chat_processor = vl_chat_processor
        self.dataset_cfg = dataset_cfg

        self.file = os.path.join(dataset_cfg.data_dir, dataset_cfg.file)

        self.data = json.load(open(self.file))

    def map(self, *args, **kargs) -> Self:
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict[str, Any]:
        data = self.data[index]["conversations"]

        # system, input_chosen = data[0], data[1]
        input_chosen = data[0]
        input_neg = list(filter(lambda x: x["role"] == "User" and x["type"] == "rejected", data))[0]
        output_chosen = list(filter(lambda x: x["role"] == "Assistant" and x["type"] == "chosen", data))[0]
        rejected = list(filter(lambda x: x["role"] == "Assistant" and x["type"] == "rejected", data))
        prompt = input_chosen

        # raw_prompt = [input_chosen, {"role": "Assistant", "content": ""}]
        chosen_data = [input_chosen, output_chosen]
        rejected_data = [[prompt, reject] for reject in rejected]
        pil_images = load_pil_images(data)

        # self.chat_processor.system_prompt = system["content"]
        # prompt_prepare = self.chat_processor(conversations=raw_prompt, images=pil_images, force_batchify=False)
        chosen_prepare = self.chat_processor(conversations=chosen_data, images=[pil_images[0]], force_batchify=False)
        rejected_prepare = [
            self.chat_processor(conversations=rejected_dt, images=[pil_images[0]], force_batchify=False)
            for rejected_dt in rejected_data
        ]
        # 输入增加负样本
        input_neg_data = [input_neg, output_chosen]
        input_neg_prepare = self.chat_processor(
            conversations=input_neg_data, images=load_pil_images(input_neg_data), force_batchify=False
        )

        history_action_input_ids = [
            self.chat_processor.tokenizer.encode(action) for action in input_chosen["history"]["actions"]
        ]

        chosen_prepare["history"] = {
            "images": self.chat_processor.image_processor(
                load_pil_images_from_path(input_chosen["history"]["images"]), return_tensors="pt"
            ).pixel_values,
            "actions": history_action_input_ids,
        }
        for rejected in rejected_prepare:
            rejected["history"] = chosen_prepare["history"]

        history_action_input_ids = [
            self.chat_processor.tokenizer.encode(action) for action in input_neg["history"]["actions"]
        ]

        input_neg_prepare["history"] = {
            "images": self.chat_processor.image_processor(
                load_pil_images_from_path(input_neg["history"]["images"]), return_tensors="pt"
            ).pixel_values,
            "actions": history_action_input_ids,
        }
        return {
            # "prompt_prepare": prompt_prepare,
            "chosen_prepare": chosen_prepare,
            "rejected_prepare": rejected_prepare,
            "input_neg_prepare": input_neg_prepare,
        }


class DPODataCollator(DPODataCollatorWithPadding):
    vl_chat_processor: VLChatProcessor

    def __init__(self, processor: VLChatProcessor):
        self.vl_chat_processor = processor
        super().__init__()

    def __call__(self, batch) -> dict[str, Any]:
        # prompt_prepares = [sample["prompt_prepare"] for sample in batch]  # B*[]
        chosen_prepares = [sample["chosen_prepare"] for sample in batch]  # B*[prepare]
        input_neg_prepares = [sample["input_neg_prepare"] for sample in batch]
        # B*[3*[prepare]]
        rejected_prepares = [sample["rejected_prepare"] for sample in batch]
        for bs in range(len(rejected_prepares)):
            rejected_prepares[bs].append(input_neg_prepares[bs])
        # rejected_prepares.extend(input_neg_prepares)
        # rejected_neg_num = len(rejected_prepares[0])
        # bs = len(batch)
        # prompts = [sample.sft_format for sample in prompt_prepares]

        # prompt_batch = self.vl_chat_processor.batchify(prompt_prepares)  # B
        chosen_batch = self.vl_chat_processor.batchify(chosen_prepares)  # B
        # input_neg_batch = self.vl_chat_processor.batchify(input_neg_prepares)
        # flatten List[List]
        rejected_prepares = list(itertools.chain(*rejected_prepares))
        rejected_batch = self.vl_chat_processor.batchify(rejected_prepares)  # B* (3+input_neg)
        return {
            # ====chosen response=====
            "chosen_input_ids": chosen_batch.input_ids,
            "chosen_attention_mask": chosen_batch.attention_mask,
            "chosen_labels": chosen_batch.labels,
            # ====rejected response=====
            "rejected_input_ids": rejected_batch.input_ids,
            "rejected_attention_mask": rejected_batch.attention_mask,
            "rejected_labels": rejected_batch.labels,
            # ===== image ====
            "pixel_values": chosen_batch.pixel_values,
            "reject_pixel_values": rejected_batch.pixel_values,
            # "chosen_images_seq_mask": chosen_batch.images_seq_mask,
            "chosen_images_emb_mask": chosen_batch.images_emb_mask,
            # "rejected_images_seq_mask": rejected_batch.images_seq_mask,
            "rejected_images_emb_mask": rejected_batch.images_emb_mask,
            "history_chosen": chosen_batch["history"],
            "history_rejected": rejected_batch["history"],
        }


def make_action_dpo_data_modlue(vl_chat_processor: VLChatProcessor, dataset_cfg: BaseDatasetConfig) -> dict[str, Any]:
    """Make dataset and collator for dpo."""
    dataset = ActionDPODataset(vl_chat_processor, dataset_cfg)
    data_collator = DPODataCollator(vl_chat_processor)

    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
