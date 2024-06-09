import logging

import evaluate
import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
)

from dataset import train_test_split
from model.callback import LoggerLogCallback
from utils import image_bytes2PIL

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(eval_preds):
    metric = "accuracy"
    accuracy = evaluate.load(metric, cache_dir="./data/evaluate")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    score = accuracy.compute(predictions=predictions, references=labels)
    return score


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    data_cfg = cfg["dataset"]

    dataset = load_dataset(data_cfg["repo"], cache_dir=data_cfg["cache_dir"])
    our_dataset = train_test_split(dataset, test_radio=0.2)

    labels = dataset["train"].unique("label")

    label2id = {c: idx for idx, c in enumerate(labels)}
    id2label = dict(enumerate(labels))

    processor = AutoImageProcessor.from_pretrained(cfg["model"]["model_path"])

    def transforms(batch):
        batch["pixel_values"] = [
            image_bytes2PIL(x["bytes"]).convert("RGB") for x in batch["image"]
        ]
        inputs = processor(batch["pixel_values"], do_resize=False, return_tensors="pt")
        inputs["labels"] = [label2id[y] for y in batch["label"]]
        return inputs

    processed_dataset = our_dataset.with_transform(transforms)

    model = ViTForImageClassification.from_pretrained(
        cfg["model"]["model_path"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    num_params = sum([p.numel() for p in model.parameters()])

    # freeze all but classifier
    for name, p in model.named_parameters():
        p.requires_grad = False

        if "classifier" in name or "vit.encoder.layer.11.attention" in name:
            p.requires_grad = True

    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    logger.info(f"{num_params = :,} | {trainable_params = :,}")

    training_args = TrainingArguments(
        run_name=cfg["run_name"],
        output_dir=f"{cfg['ckpt_dir']}/{cfg['run_name']}",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        **cfg["training"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=processor,  #  make sure the image processor configuration file (stored as JSON) will also be uploaded to the repo on the Hub.
    )
    trainer.add_callback(LoggerLogCallback(logger))

    trainer.train()

    logger.info(trainer.evaluate(processed_dataset["test"]))

    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
