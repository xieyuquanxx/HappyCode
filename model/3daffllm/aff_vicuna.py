import logging
import math

# import einops
import os
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import yaml
from common.registry import registry
from common.utils import get_rank, get_world_size
from easydict import EasyDict
from gorilla.config import Config
from models.base_model import BaseModel, disabled_train
from models.modeling_llama import LlamaForCausalLM
from models.openad.utils import *
from models.pointbert.point_encoder import PointTransformer
from models.seg_utils.loss import dice_loss, sigmoid_ce_loss
from packaging import version

# from transformers import LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from torch.cuda.amp import autocast as autocast
from transformers import LlamaTokenizer, LlamaTokenizerFast

from .segment_anything import (
    build_aff_openad,
    build_aff_openad_output,
    build_aff_openad_output_train,
)

SEG_TOKEN = "<SEG>"


# The releated Function with Load PointBert
def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    print("point类型", cfg_file)
    config = EasyDict()
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class AffordanceVicunaOpenAD(BaseModel):
    """
    Comm Vicuna model.
    Supported model types:
        - Vicuna7b
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "/workspace/project/Aff_LLM_debug/configs/models/point_vicuna7b.yaml",
    }

    def __init__(
        self,
        mix_precision="bf16",
        # The seg point feature dimension
        prompt_encoder_dim=512,
        # loss设置
        ce_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
        # Point_encoder设置,新添加的参数
        point_model_config_path=None,
        freeze_point=True,
        # seg_encoder设置
        free_seg_point_encoder=True,
        seg_point_encoder_config_path=None,
        seg_point_encoder_path=None,
        # aff_decoder
        aff_path=None,
        train_aff_decoder=False,
        upscale_points=2048,
        # Lora设置
        lora_r=8,
        lora_alpha=16,
        # LLM 设置
        llm_model=None,
        freeze_llm=True,
        prompt="",
        max_txt_len=128,
        max_output_txt_len=128,
        # apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        freeze_linear=True,
        lora_llm_finetune=False,
        **kwargs,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse(
            "4.28"
        ), "BLIP-2 Vicuna requires transformers>=4.28"
        # Set Precision
        self.mix_precision = mix_precision
        # Point_Encoder，ULIP2_PointBert initlize Point_encoder
        self.point_model_config = cfg_from_yaml_file(point_model_config_path)
        self.point_encoder = PointTransformer(self.point_model_config.model)
        self.point_encoder.load_checkpoint(self.point_model_config.model_path)
        # The PointSeg_encoder
        # OpenAD_modify
        openadcfg = Config.fromfile(seg_point_encoder_config_path)
        self.seg_point_encoder = build_model_checkpoint(
            openadcfg, seg_point_encoder_path, is_eval=True
        )
        self.upscale_points = upscale_points
        # loss weight
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight

        # The PointEncoder before the LLM
        if freeze_point:
            for name, param in self.point_encoder.named_parameters():
                param.requires_grad = False
            self.point_encoder = self.point_encoder.eval()
            self.point_encoder.train = disabled_train
            print("freeze point encoder")
            logging.info("freeze point encoder")

        # The Point SegEncoder in Affordance predict (the upper line in model structure picture)
        if free_seg_point_encoder:
            for name, param in self.seg_point_encoder.named_parameters():
                param.requires_grad = False
            self.seg_point_encoder = self.seg_point_encoder.eval()
            self.seg_point_encoder.train = disabled_train
            print("freeze seg_point encoder")
            logging.info("freeze seg_point encoder")
        # LLM
        print("Start Load LLM Checkpoint")
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            llm_model, use_fast=False, truncation_side="left"
        )
        memory = {}
        for i in range(get_world_size()):
            if i == get_rank():
                memory[i] = "80GiB"
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=memory,
        )
        print("Load Original LLM Model Successfully")
        if freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            print("freeze LLM")
            logging.info("freeze LLM")

        # 添加Lora
        if lora_llm_finetune:
            print("using lora llm finetune")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                # target_modules=[
                #     "q_proj", "k_proj", "v_proj", "o_proj",
                #     "gate_proj", "down_proj", "up_proj"
                # ],
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()

        # Add special seg token after the Lora setting
        # self.llm_tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
        self.llm_tokenizer.add_tokens([SEG_TOKEN])
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.seg_token_id = self.llm_tokenizer.convert_tokens_to_ids(SEG_TOKEN)

        # points to llm projector,简单的直接线性映射
        self.llm_proj = nn.Linear(
            self.point_model_config.model.trans_dim, self.llm_model.config.hidden_size
        )
        if freeze_linear:
            for name, param in self.llm_proj.named_parameters():
                param.requires_grad = False
            self.llm_proj = self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            print("freeze point encoder to LLM liner")
        # # Calculate the total number of trainable parameters
        # llm_proj_total_params = sum(p.numel() for p in self.llm_proj.parameters() if p.requires_grad)
        # print(f"The llm_projection layer has a total of {llm_proj_total_params} trainable parameters.")

        # aff_decoder model and projector
        self.prompt_encoder_dim = prompt_encoder_dim
        self.aff_model, self.aff_proj = self.initialize_affordance_modules(
            aff_path,
            in_dim=self.llm_model.config.hidden_size,
            out_dim=self.prompt_encoder_dim,
            train_aff_decoder=train_aff_decoder,
        )
        # #Calculate the total parameters number of aff_model
        # print("aff_model参数量")
        # self.aff_model.counting_training_parameters()

        # # Calculate the total number of aff_proj trainable parameters
        # aff_proj_total_params = sum(p.numel() for p in self.aff_proj.parameters() if p.requires_grad)
        # print(f"The aff_projection layer has a total of {aff_proj_total_params} trainable parameters.")

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        # self._apply_lemmatizer = apply_lemmatizer
        # self._lemmatizer = None
        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.counting_training_parameters()

    def initialize_affordance_modules(
        self, aff_path, in_dim, out_dim, train_aff_decoder=True
    ):
        # aff_model = build_aff_openad(aff_path)
        aff_model = build_aff_openad_output(aff_path)
        # aff_model = build_aff_openad_output_train()

        if train_aff_decoder:
            aff_model.train()
            for param in aff_model.parameters():
                param.requires_grad = True
        # print(type(in_dim))
        # Projection layer
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        # aff_proj = nn.ModuleList([nn.Sequential(*text_fc)])
        aff_proj = nn.Sequential(*text_fc)
        aff_proj.train()
        for param in aff_proj.parameters():
            param.requires_grad = True

        return aff_model, aff_proj

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(
                torch.cat(
                    [
                        input_ids[i][:this_input_ones],
                        output_ids[i][1:],
                        input_ids[i][this_input_ones:],
                    ]
                )
            )
            llm_tokens["attention_mask"].append(
                torch.cat(
                    [
                        input_atts[i][:this_input_ones],
                        output_atts[i][1:],
                        input_atts[i][this_input_ones:],
                    ]
                )
            )
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    def counting_training_parameters(self):
        total = 0.0
        trainable_names = []
        all = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total += param.nelement()
                trainable_names.append(name)
            all += param.nelement()
        print(trainable_names)
        print("  + Number of trainable params: %.2fM" % (total / 1e6))
        print("Number of all params: %.2fM" % (all / 1e6))
        return total

    # The point_encoder before LLM
    def encode_point(self, points):
        # with self.maybe_autocast(torch.bfloat16):
        with self.maybe_autocast(self.mix_precision):
            # The input data is a list
            points2bs = torch.stack(points)
            # points_feat,b*513*384  ,points_pos,2*513*384
            # Include the cls Token
            points_feat, points_pos = self.point_encoder(points2bs)
            inputs_llm = self.llm_proj(points_feat)

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
            points2bs.device
        )
        return inputs_llm, atts_llm, points_pos

    # pred mask
    def predict_mask(
        self,
        output_ids,
        last_hidden_states,
        points,
        shape_id,
        original_list=(1, 2048),
    ):
        """
        postprocess_masks()函数

        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        original_list = (1, self.upscale_points)

        if points is None:
            return []

        points2bs = torch.stack(points)
        points2bs = points2bs.transpose(1, 2)
        point_embedding = self.seg_point_encoder(points2bs)
        # Debug the point features
        if torch.isnan(point_embedding).any():
            print("point_embedding from segPointencoder is nan")
            print("shape id", shape_id)
            raise ValueError("src from image_embedding is nan")

        projected_hidden_state = self.aff_proj(
            last_hidden_states
        )  # [bs, length, dim],2*40*512

        pred_masks = []
        for i in range(projected_hidden_state.shape[0]):
            seg_token_mask = output_ids[i][:] == self.seg_token_id

            if seg_token_mask.sum() == 0:
                pred_masks.append(
                    torch.zeros((1, *original_list), dtype=torch.float32).cuda()
                )
                continue

            seq_length = last_hidden_states.shape[1]
            seg_token_mask = seg_token_mask[:seq_length]

            pred_embeddings_ = projected_hidden_state[i][
                seg_token_mask
            ]  # [num_seg, dim],示例40*512-->1*512,只有1个seg
            point_embedding_ = point_embedding[i].unsqueeze(0)

            # point_openad set
            pred_mask = self.aff_model(
                image_embeddings=point_embedding_,
                image_pe=point_embedding_,
                sparse_prompt_embeddings=pred_embeddings_.unsqueeze(1),
                multimask_output=False,
            )
            pred_masks.append(pred_mask)

        return pred_masks

    def add_prompt(self, history, question, no_prompt=True):
        if no_prompt:
            return question
        # history: [(q1, a1), (q2, a2), ...]. dose not include current QA
        prompt = ""
        for q, a in history:
            prompt += "USER: " + q + "\n"
            prompt += "ASSISTANT: " + a + "</s>\n"

        prompt += "USER: " + question + "\n"
        prompt += "ASSISTANT: "
        return prompt

    def forward(self, samples):
        history = samples.get("history", None)
        questions = samples["question"]
        answers = samples["answer"]

        bs = len(questions)
        history = [[]] * bs if history is None else history
        new_questions = []
        for i in range(bs):
            res = self.add_prompt(history[i], questions[i], no_prompt=False)
            new_questions.append(res)

        questions = new_questions

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            questions,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in answers],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        attention_mask = llm_tokens["attention_mask"]

        # with self.maybe_autocast(torch.bfloat16):
        with self.maybe_autocast(self.mix_precision):
            if "points" in samples:
                points = samples["points"]

                inputs_llm, atts_llm, point_pos_em = self.encode_point(
                    points
                )  # 此时atts_llm为b*513

                # do not apply loss to the query tokens
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long)
                    .to(inputs_llm.device)
                    .fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)

                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True,
            )

            # predict masks
            points = samples.get("points", None)
            gt_masks = samples.get("masks", None)
            shape_id = samples.get("shape_id", None)
            # print("gt_mask_shape",gt_masks.shape)

            if gt_masks is None:
                return {
                    "loss": outputs.loss,
                    "ce_loss": outputs.loss,
                    "mask_bce_loss": 0,
                    "mask_dice_loss": 0,
                    "mask_loss": 0,
                }

            # remember to drop image token
            image_token_length = inputs_llm.shape[1]
            last_hidden_states = outputs.hidden_states[-1][:, image_token_length:, :]

            # note that there is an offset of 1 between output_ids and hidden_states
            pad_token_id = self.llm_tokenizer.pad_token_id
            output_ids = llm_tokens["input_ids"][:, 1:]
            output_ids = torch.concat(
                [
                    output_ids,
                    torch.full((output_ids.shape[0], 1), pad_token_id)
                    .to(torch.long)
                    .cuda(),
                ],
                dim=1,
            )

            pred_masks = self.predict_mask(
                output_ids,
                last_hidden_states,
                points,
                shape_id,
            )

        # loss
        ce_loss = outputs.loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            # Load 3d Affordance Net data, use the loss of 3d affordance_net
            # mask_bce_loss += (
            #         sigmoid_ce_loss_affordance_net(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
            #         * gt_mask.shape[0]
            # )
            # mask_dice_loss += (
            #         dice_loss_affordance_net(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
            #         * gt_mask.shape[0]
            # )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss
        # Debugging code
        if any(
            math.isnan(loss.item())
            for loss in (ce_loss, mask_bce_loss, mask_dice_loss, total_loss)
        ):
            print("One or more losses are NaN!")
            print("ce_loss:", ce_loss)
            print("mask_bce_loss:", mask_bce_loss)
            print("mask_dice_loss:", mask_dice_loss)
            print("total_loss:", total_loss)

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        history = samples.get("history", None)
        questions = samples["question"]

        if isinstance(questions, str):
            questions = [questions]

        bs = len(questions)
        history = [[]] * bs if history is None else history
        new_questions = []
        for i in range(bs):
            res = self.add_prompt(history[i], questions[i], no_prompt=False)
            new_questions.append(res)

        questions = new_questions

        self.llm_tokenizer.padding_side = "left"
        llm_tokens = self.llm_tokenizer(
            questions, padding="longest", return_tensors="pt"
        ).to(self.device)

        # with self.maybe_autocast(torch.bfloat16):
        with self.maybe_autocast(self.mix_precision):
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            attention_mask = llm_tokens.attention_mask

            if "points" in samples:
                points = samples["points"]
                masks = samples.get("masks", None)
                shape_id = samples.get("shape_id", None)
                inputs_llm, atts_llm, points_pos = self.encode_point(points)

                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                # num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                # num_return_sequences=num_captions,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
            )

            pred_masks = []
            if "points" in samples:
                output_ids = outputs.sequences[:, 1:-1]  # remember to drop bos token
                last_hidden_states = [
                    token_states[-1] for token_states in outputs.hidden_states[1:]
                ]  # remember to drop input token

                if len(last_hidden_states) == 0:
                    return {"text": "", "masks": []}

                last_hidden_states = torch.concat(last_hidden_states, dim=1)

                pred_masks = self.predict_mask(
                    output_ids,
                    last_hidden_states,
                    points,
                    shape_id,
                )

        try:
            output_text = self.llm_tokenizer.batch_decode(
                outputs["sequences"], skip_special_tokens=True
            )
        except Exception as e:
            print(outputs)
            raise e
        output_text = [text.strip() for text in output_text]
        masks_score = [score.sigmoid() for score in pred_masks]
        # print("mask_score",masks_score)
        pro_pred_masks = [(m > 0).to(torch.float32) for m in pred_masks]
        output_dict = {
            "text": output_text,
            "masks_scores": masks_score,
            "masks": pro_pred_masks,
            "output_ids": output_ids,
            "attentions": outputs.attentions,
            "seg_id": self.seg_token_id,
        }

        return output_dict

    # It is necessary to set different parameters with different learning rates.

    # def get_optimizer_params(self, weight_decay, lr_scale=1):
    #     # Returns different parts of the parameters, possibly with different learning rates and weight decay
    #     p_wd, p_non_wd = [], []
    #     aff_param = []
    #     for n, p in self.named_parameters():
    #         if not p.requires_grad:
    #             continue  # frozen weights
    #         if "aff_model" in n:
    #             aff_param.append(p)
    #         elif p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
    #             p_non_wd.append(p)
    #         else:
    #             p_wd.append(p)
    #     main_optim_params = [
    #         {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
    #         {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
    #     ]
    #     aff_optim_params=[{"params": aff_param,"weight_decay": weight_decay, "lr_scale": lr_scale}]
    #     return main_optim_params,aff_optim_params

    # def get_optimizer_params(self, weight_decay, init_lr, aff_lr, aff_weight_decay, lr_scale=1):
    #     # Returns different parts of the parameters, possibly with different learning rates and weight decay
    #     p_wd, p_non_wd = [], []
    #     aff_param = []
    #     for n, p in self.named_parameters():
    #         if not p.requires_grad:
    #             continue  # frozen weights
    #         if "aff_model" in n:
    #             aff_param.append(p)
    #         elif p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
    #             p_non_wd.append(p)
    #         else:
    #             p_wd.append(p)
    #     optim_params = [
    #         {"params": p_wd, "lr": init_lr, "weight_decay": weight_decay, "lr_scale": lr_scale},
    #         {"params": p_non_wd, "lr": init_lr, "weight_decay": 0, "lr_scale": lr_scale},
    #         {"params": aff_param, "lr": aff_lr, "weight_decay": aff_weight_decay, "lr_scale": lr_scale},
    #     ]
    #     return optim_params

    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_config(cls, cfg):
        # Set the mix precision in forward and generate
        mix_precision = cfg.get("mix_precision", "bf16")
        # The seg point Feature dimension
        prompt_encoder_dim = cfg.get("prompt_encoder_dim", 512)
        # Point_encoder set
        point_model_config_path = cfg.get(
            "point_model_config_path",
            "/workspace/project/Aff_LLM_debug/configs/models/PointTransformer_2048point.yaml",
        )
        freeze_point = cfg.get("freeze_point", True)
        # seg_encoder
        free_seg_point_encoder = cfg.get("free_seg_point_encoder", False)
        seg_point_encoder_config_path = cfg.get(
            "seg_point_encoder_config_path",
            "/workspace/project/Aff_LLM_debug/models/openad/config/full_shape_cfg_DGCNN_modify.py",
        )
        seg_point_encoder_path = cfg.get(
            "seg_point_encoder_path",
            "/workspace/project/Aff_LLM_debug/model_ckpt/DGCNN_best_model.t7",
        )
        # aff_decoder
        aff_path = cfg.get("aff_path", None)
        train_aff_decoder = cfg.get("train_aff_decoder", False)
        upscale_points = 2048
        # Lora
        lora_r = cfg.get("lora_r", 8)
        lora_alpha = cfg.get("lora_alpha", 16)
        # LLM
        llm_model = cfg.get("llm_model", None)
        freeze_llm = cfg.get("freeze_llm", True)
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 128)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        freeze_linear = cfg.get("freeze_linear", False)
        lora_llm_finetune = cfg.get("lora_llm_finetune", False)

        model = cls(
            mix_precision=mix_precision,
            prompt_encoder_dim=prompt_encoder_dim,
            point_model_config_path=point_model_config_path,
            freeze_point=freeze_point,
            free_seg_point_encoder=free_seg_point_encoder,
            seg_point_encoder_config_path=seg_point_encoder_config_path,
            seg_point_encoder_path=seg_point_encoder_path,
            aff_path=aff_path,
            train_aff_decoder=train_aff_decoder,
            upscale_points=upscale_points,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            llm_model=llm_model,
            freeze_llm=freeze_llm,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            freeze_linear=freeze_linear,
            lora_llm_finetune=lora_llm_finetune,
        )

        model.load_checkpoint_from_config(cfg)

        return model
