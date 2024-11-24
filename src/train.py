import os
from pathlib import Path
import logging
import comet_ml

import hydra
import torch
import lightning as L
from lightning.pytorch.loggers import Logger 
from typing import List
from omegaconf import DictConfig
from dotenv import load_dotenv
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)
load_dotenv("../.env")

class CustomModelCheckpiont(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        trainer.lightning_module.save_transformed_model = True
        filepath = filepath.split(".ckpt")[0]
        filepath = f"{filepath}_patch_size-{trainer.model.patch_size}_embed_dim-{trainer.model.embed_dim}.ckpt"
        print(f"File saved to {filepath}")
        super()._save_checkpoint(trainer, filepath)
        # print(filepath)


def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks
    
    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    # print("callbacks-", callbacks)
    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers
    
    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

@task_wrapper
def train_task(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting training!")
    print("#######training started####")
    trainer.fit(model, datamodule)
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")
    return train_metrics


@task_wrapper
def test_task(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule
):
    log.info("Starting testing!")
    checkpoint_curr_best_model_path = trainer.checkpoint_callback.best_model_path.split(".ckpt")[0]
    checkpoint_curr_best_model_path = f"{checkpoint_curr_best_model_path}_patch_size-{trainer.model.patch_size}_embed_dim-{trainer.model.embed_dim}.ckpt"
    print(f"File saved to {checkpoint_curr_best_model_path}")
    log.info(f"test Check point {checkpoint_curr_best_model_path}")
    if checkpoint_curr_best_model_path:
        log.info(
            f"Loading best checkpoint: {checkpoint_curr_best_model_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=checkpoint_curr_best_model_path
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)
    log.info(f"Test metrics:\n{test_metrics}")
    return  test_metrics[0] if test_metrics else test_metrics

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    print("cfg")
    print(cfg)
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    # print("datamodule")
    # print(len(datamodule.train_dataloader()))
    # print(len(datamodule.train_dataloader()))

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    train_metrics = {}

    # Train the model
    if cfg.get("train"):
        train_metrics = train_task(cfg, trainer, model, datamodule)

    test_metrics = {}
    # Test the model
    if cfg.get("test"):
        test_metrics = test_task(cfg, trainer, model, datamodule)

    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    # all_metrics = {**train_metrics}

    # Extract and return the optimization metric
    optimization_metric = all_metrics.get(cfg.get("optimization_metric"))
    if optimization_metric is None:
        log.warning(f"Optimization metric '{cfg.get('optimization_metric')}' not found in metrics. Returning 0.")
        return 0.0
    
    return optimization_metric


if __name__ == "__main__":
    train()