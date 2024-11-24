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

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)
load_dotenv("../.env") 

def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks
    
    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
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
def test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule
):
    test_metrics = {}
    log.info("Starting testing!")
    print(f"checkpoint path file {cfg.eval.checkpoint_path_file}")
    optimized_checkpoint_filename = ""
    with open(cfg.eval.checkpoint_path_file, 'r') as file:
        optimized_checkpoint_filename = file.readline().strip() 

    if optimized_checkpoint_filename:
        log.info(
            f"Loading best checkpoint: {optimized_checkpoint_filename}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=optimized_checkpoint_filename
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)
    log.info(f"Test metrics:\n{test_metrics}")
    return  test_metrics[0] if test_metrics else test_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def eval(cfg: DictConfig):
    # print(cfg.get("optimization_metric"))
    # exit()
    log_dir = Path(cfg.paths.log_dir)
    
    setup_logger(log_dir/"eval_log.log")

    # Set seed for reproducibility
    seed_everything(42)
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")


    # Initialize the data module
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    # data_module = DogBreedImageDataModule(dl_path=data_dir)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )
    # print(cfg)
    # Train the model
    test_metrics = {}

    if cfg.get("test"):
        test_metrics = test(cfg, trainer, model, datamodule)

    all_metrics = {**test_metrics}

    # Extract and return the optimization metric
    optimization_metric = all_metrics.get(cfg.get("optimization_metric"))
    print("optimization_metric-", cfg.get("optimization_metric"), optimization_metric, all_metrics)
    if optimization_metric is None:
        log.warning(f"Optimization metric '{cfg.get('optimization_metric')}' not found in metrics. Returning 0.")
        return 0.0
    
    return optimization_metric


if __name__ == "__main__":
    eval()