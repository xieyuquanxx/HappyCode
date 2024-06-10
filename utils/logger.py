import logging
import os

from omegaconf import DictConfig
from rich.logging import RichHandler


def get_logger(name: str, cfg: DictConfig) -> logging.Logger:
    log_dir = cfg["log"]["dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, cfg["log"]["file"])
    file_handler = logging.FileHandler(log_file)
    rich_handler = RichHandler(markup=True)

    logging.basicConfig(
        format="[%(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level="INFO"
    )
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False

    return log
