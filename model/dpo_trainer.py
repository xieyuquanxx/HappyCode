from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer as Optimizer
from trl import DPOTrainer


class VLDPOTrainer(DPOTrainer):
    def concatenated_inputs(
        self,
        batch: dict[str, torch.Tensor],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int | None = 0,
        device: torch.device | None = None,
    ) -> dict[str, torch.LongTensor]:
        # if padding_value is None:
        #     padding_value = 0
        concatenated_batch = super().concatenated_inputs(
            batch, is_encoder_decoder, label_pad_token_id, padding_value, device
        )
        if "pixel_values" in batch:
            # duplicate img_input in batchsize dimension
            # v, rejected_neg_number = batch["pixel_values"], batch["rejected_input_ids"].shape[0]
            # bs, pixel_shape = v.shape[0], v.shape[1:]
            # rejected_neg_number //= bs  # =3
            # b = v.repeat(1, rejected_neg_number, 1, 1, 1).reshape(bs * rejected_neg_number, *pixel_shape)

            # # input_neg sample concatenated
            # input_neg_images = batch["input_neg_pixel_values"]

            # concatenated_batch["concatenated_pixel_values"] = torch.cat([v, b, input_neg_images], dim=0)  # type: ignore
            concatenated_batch["concatenated_pixel_values"] = torch.cat(
                [batch["pixel_values"], batch["reject_pixel_values"]], dim=0
            )
            # if "chosen_images_seq_mask" in batch:
            concatenated_batch["concatenated_images_seq_mask"] = (
                concatenated_batch["concatenated_input_ids"] == 100015
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
        self, model: Module, batch: dict[str, list | torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # torch.cuda.empty_cache()
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
        all_logits = output.logits  # [chosen B+ rejected B + input neg B, Length, 102400]
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

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_logps_avg,
        )

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            neg_number = policy_rejected_logps.shape[0]
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            if neg_number > 1:
                # todo: anchor loss design
                losses = losses + (reference_chosen_logps - policy_chosen_logps)
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device)
                - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        bs = policy_chosen_logps.shape[0]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps.reshape(bs, -1),
            policy_rejected_logps.reshape(bs, -1),
            reference_chosen_logps.reshape(bs, -1),
            reference_rejected_logps.reshape(bs, -1),
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards.reshape(bs, -1) - rejected_rewards.reshape(bs, -1)).mean().cpu()
        )
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics
