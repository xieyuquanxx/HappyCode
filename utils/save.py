import torch
import transformers


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str
) -> None:
    """Collects the state dict and dump to disk.
    https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py#L185
    """

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
