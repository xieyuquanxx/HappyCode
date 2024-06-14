import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


@contextmanager
def event_time(event_name: str):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    inference_time = start.elapsed_time(end)

    logger.info(f"{event_name}: {inference_time:.2f}ms")
    return inference_time  # ms
