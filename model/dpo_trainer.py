from abc import ABC
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer as Optimizer
from transformers import (
    PreTrainedModel,
    Trainer,
)
from trl import DPOTrainer


def pad_to_length(
    tensor: torch.Tensor,
    length: int,
    pad_value: Union[int, float],
    dim: int = -1,
    padding_side: Literal["right", "left"] = "right",
) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        if padding_side == "right":
            return torch.cat(
                [
                    tensor,
                    pad_value
                    * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )
        elif padding_side == "left":
            return torch.cat(
                [
                    pad_value
                    * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                    tensor,
                ],
                dim=dim,
            )
        else:
            raise ValueError(f"Unknown padding_side: {padding_side}")


class VLDPOTrainer(DPOTrainer, ABC):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        self.finetuning_args = finetuning_args
        self.processor = processor
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = -100
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False)
                    or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
                self.ref_model.eval()

    def tokenize_row(
        self, feature, model: PreTrainedModel | Module | None = None
    ) -> Dict:
        assert self.processor is not None, "Processor must be provided."
        # <image_placeholder>xxxx
        prompt = feature["prompt"]
        # prompt = self.processor.format_multimodal_prompt(
        #     feature["prompt"], feature["img_path"]
        # )  # add image placeholder to prompt
        prompt = self.processor.make_single_turn_conv(
            prompt, ""
        )  # This returns [{"user":<image>\n<prompt>,"assistant":""}]
        prompt = self.processor.process_batch_conv(
            [prompt], system_message=None, add_end_for_empty_value=False
        )  # This returns {"prompt":None,"answer":None,"full":None,"raw_str":[...]}
        prompt_raw_str = prompt["raw_str"][
            0
        ]  # This returns "USER: <image>\n<prompt> ASSISTANT:"
        assistant_end = self.processor.chat_template.sep2
        feature["chosen"] += assistant_end
        feature["rejected"] += assistant_end
        feature["prompt"] = prompt_raw_str
        batch = super().tokenize_row(feature, model)
        batch["img_path"] = feature["img_path"]
        return batch

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = super().concatenated_inputs(
            batch, is_encoder_decoder, label_pad_token_id, padding_value, device
        )
        if "pixel_values" in batch:
            # duplicate img_input in batchsize dimension
            v = batch["pixel_values"]
            concatenated_batch["concatenated_pixel_values"] = torch.cat([v, v], dim=0)  # type: ignore
            concatenated_batch["concatenated_images_seq_mask"] = (
                concatenated_batch["input_ids"] == 100015
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

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        mask_shared_tokens: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        if mask_shared_tokens:
            len_chosen = labels.shape[0] // 2
            assert len_chosen * 2 == labels.shape[0]
            chosen_labels = labels[:len_chosen]
            rejected_labels = labels[len_chosen:]
            chosen_shared_mask = torch.full_like(chosen_labels, False, dtype=torch.bool)
            rejected_shared_mask = torch.full_like(
                rejected_labels, False, dtype=torch.bool
            )
            min_match_size = 3
            for idx, (chosen_label, rejected_label) in enumerate(
                zip(chosen_labels, rejected_labels)
            ):
                c_mod, r_mod = get_diff_ids(
                    chosen_label.tolist(),
                    rejected_label.tolist(),
                    min_match_size=min_match_size,
                )
                chosen_shared_mask[idx][c_mod] = True
                rejected_shared_mask[idx][r_mod] = True
            shared_mask = torch.cat([chosen_shared_mask, rejected_shared_mask], dim=0)
            loss_mask &= shared_mask
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
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
            model_kwargs.update(
                {"images_seq_mask": concatenated_batch["concatenated_images_seq_mask"]}
            )
        if "concatenated_images_emb_mask" in concatenated_batch:
            model_kwargs.update(
                {"images_emb_mask": concatenated_batch["concatenated_images_emb_mask"]}
            )
        if "concatenated_pixel_values" in concatenated_batch:
            model_kwargs.update(
                {"pixel_values": concatenated_batch["concatenated_pixel_values"]}
            )

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
        all_logps = self.get_batch_logps(
            all_logits,
            final_labels,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            mask_shared_tokens=self.loss_type == "ddpo",
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor(
                [0], dtype=pi_logratios.dtype, device=pi_logratios.device
            )
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid" or self.loss_type == "ddpo":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            chosen_KL = (
                (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            )
            rejected_KL = (
                (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
            )

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device)
                - reference_chosen_logps.to(self.accelerator.device)
            ).detach()  # noqa
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()  # noqa
        )

        return losses, chosen_rewards, rejected_rewards

    # def get_batch_samples(
    #     self, model, batch: Dict[str, torch.LongTensor]
    # ) -> Tuple[str, str]:
    #     """Generate samples from the model and reference model for the given batch of inputs."""

    #     # If one uses `generate_during_eval` with peft + bf16, we need to explictly call generate with
    #     # the torch cuda amp context manager as some hidden states are silently casted to full precision.
    #     generate_context_manager = (
    #         nullcontext
    #         if not self._peft_has_been_casted_to_bf16
    #         else torch.cuda.amp.autocast
    #     )

    #     others = {}
    #     if "img_input_dict" in batch:
    #         others.update(batch["img_input_dict"])
    #     with generate_context_manager():
    #         policy_output = model.generate(
    #             input_ids=batch["prompt_input_ids"],
    #             attention_mask=batch["prompt_attention_mask"],
    #             max_length=self.max_length,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             **others,
    #         )

    #         # if reference_output in batch use that otherwise use the reference model
    #         if "reference_output" in batch:
    #             reference_output = batch["reference_output"]
    #         else:
    #             if self.ref_model is None:
    #                 with self.null_ref_context():
    #                     reference_output = self.model.generate(
    #                         input_ids=batch["prompt_input_ids"],
    #                         attention_mask=batch["prompt_attention_mask"],
    #                         max_length=self.max_length,
    #                         do_sample=True,
    #                         pad_token_id=self.tokenizer.pad_token_id,
    #                         **others,
    #                     )
    #             else:
    #                 reference_output = self.ref_model.generate(
    #                     input_ids=batch["prompt_input_ids"],
    #                     attention_mask=batch["prompt_attention_mask"],
    #                     max_length=self.max_length,
    #                     do_sample=True,
    #                     pad_token_id=self.tokenizer.pad_token_id,
    #                     **others,
    #                 )

    #     policy_output = pad_to_length(
    #         policy_output, self.max_length, self.tokenizer.pad_token_id
    #     )
    #     policy_output_decoded = self.tokenizer.batch_decode(
    #         policy_output, skip_special_tokens=True
    #     )

    #     reference_output = pad_to_length(
    #         reference_output, self.max_length, self.tokenizer.pad_token_id
    #     )
    #     reference_output_decoded = self.tokenizer.batch_decode(
    #         reference_output, skip_special_tokens=True
    #     )

    #     return policy_output_decoded, reference_output_decoded
