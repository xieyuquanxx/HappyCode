import transformers


class LoggerLogCallback(transformers.TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        control.should_log = False
        if state.is_local_process_zero:
            self.logger.info(logs)
