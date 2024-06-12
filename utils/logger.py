import logging
import os

from model import LogConfig

# from rich.logging import RichHandler


def rank0_log(local_rank: int | None, logger: logging.Logger, msg: str) -> None:
    """
    Logs the given message using the provided logger if the local rank is 0 or -1.

    Args:
        local_rank (int | None): The local rank.
        logger (logging.Logger): The logger object to use for logging.
        msg (str): The message to log.
    """
    if local_rank == 0 or local_rank == -1 or local_rank is None:
        logger.info(msg)


def get_logger(name: str, cfg: LogConfig) -> logging.Logger:
    log_dir = cfg.dir
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, cfg.file)
    file_handler = logging.FileHandler(log_file)
    # rich_handler = RichHandler(markup=True)

    logging.basicConfig(
        format="[%(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level="INFO"
    )
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    # log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False

    return log
