from typing import Any
import wandb
import logging
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)

class WandBLogger:
    """
    Weights & Biases logger for continual learning experiments.

    Args:
        project_name (str): Name of the W&B project
        experiment_name (str): Name of this specific run
        config (DictConfig): Configuration to log
        tags (list, optional): Tags for the run
        api_key (str, optional): Weights & Biases API key
        entity (str, optional): W&B username/organization
        save_code (bool, optional): Whether to save code to W&B
        log_model (bool, optional): Whether to log model artifacts
    """

    def __init__(self, config: DictConfig):
        try:
            logging.info(f"Initializing W&B logger...")
            if config.logging.api_key:
                wandb.login(key=config.logging.api_key)
                logger.info("Successfully logged in to W&B using API key")
            else:
                logger.warning("No API key provided for W&B login")

            self.run = wandb.init(
                project=config.logging.project_name,
                name=config.logging.experiment_name,
                entity=config.logging.entity,
                tags=config.logging.tags,
                config=OmegaConf.to_container(config, resolve=True),  # Log the entire config # type: ignore
                reinit='finish_previous',
                settings=wandb.Settings(code_dir=".") if config.logging.save_code else None,
                job_type="inference"
            )
            self.should_log_model = config.logging.log_model
            logger.info(f"Successfully initialized W&B run: {config.logging.experiment_name}")
            print(f"W&B Run initialized. View live results on your dashboard: {self.run.url}")

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {str(e)}")
            self.run = None

    def log(self, data: dict[str, Any], step: int | None = None, commit: bool | None = None) -> None:
        """
        Log data to the W&B run.
        Args:
            data (dict): Data to log.
            step (int, optional): Step number for the logged data.
            commit (bool, optional): Whether to commit the logged data immediately.
        Returns:
            None
        """
        if self.run:
            self.run.log(data=data, step=step, commit=commit)
        else:
            logger.warning("No W&B run to log data to. Please initialize the run first.")

    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the W&B run.

        Args:
            exit_code (int): Exit code for the run
        """
        if self.run:
            self.run.finish(exit_code=exit_code)
            logger.info(f"W&B run finished with exit code {exit_code}")
        else:
            logger.warning("No W&B run to finish")