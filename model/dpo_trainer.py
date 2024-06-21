from typing import Dict, List, Optional, Tuple, Union, override

import torch
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer as Optimizer
from trl import DPOTrainer


class VLDPOTrainer(DPOTrainer):
    def concatenated_inputs(
        self,
        batch: Dict[str, torch.Tensor],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        if padding_value is None:
            padding_value = 0
        concatenated_batch = super().concatenated_inputs(
            batch, is_encoder_decoder, label_pad_token_id, padding_value, device
        )
        if "pixel_values" in batch:
            # duplicate img_input in batchsize dimension
            v = batch["pixel_values"]
            concatenated_batch["concatenated_pixel_values"] = torch.cat([v, v], dim=0)  # type: ignore
            concatenated_batch["concatenated_images_seq_mask"] = (
                concatenated_batch["concatenated_input_ids"] == 10001
            )
            if "chosen_images_emb_mask" in batch:
                concatenated_batch["concatenated_images_emb_mask"] = torch.cat(
                    [
                        batch["chosen_images_emb_mask"],
                        batch["rejected_images_emb_mask"],
                    ],
                    dim=0,
                )
        return concatenated_batch

    def concatenated_forward(
        self, model: Module, batch: Dict[str, Union[List, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {"use_cache": False}

        if "concatenated_images_seq_mask" in concatenated_batch:
            model_kwargs.update({"images_seq_mask": concatenated_batch["concatenated_images_seq_mask"].bool()})
        if "concatenated_images_emb_mask" in concatenated_batch:
            model_kwargs.update({"images_emb_mask": concatenated_batch["concatenated_images_emb_mask"].bool()})
        if "concatenated_pixel_values" in concatenated_batch:
            model_kwargs.update({"pixel_values": concatenated_batch["concatenated_pixel_values"]})

        output = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            labels=concatenated_batch["concatenated_labels"],
            **model_kwargs,
        )
        all_logits = output.logits
        if hasattr(output, "labels"):
            final_labels = output.labels
        else:
            final_labels = concatenated_batch["concatenated_labels"]
        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            final_labels,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        chosen_logps_avg = all_logps[:len_chosen] / size_completion[:len_chosen]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps_avg)
