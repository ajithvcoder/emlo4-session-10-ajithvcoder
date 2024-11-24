import os
from pathlib import Path
import logging
import comet_ml


import hydra
import torch
import lightning as L
from lightning.pytorch.loggers import Logger
from PIL import Image, ImageDraw, ImageFont
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

def annotate_images(data_list, class_names, output_path):
    for tensor, (image_path,) in data_list:
        class_id = tensor.item()
        class_name = class_names.get(class_id, 'Unknown')
        # print(image_path)
        actual_category = os.path.basename(os.path.dirname(image_path))
        try:
            image = Image.open(image_path)
        except IOError:
            print(f"Failed to load image at {image_path}")
            continue

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("./assets/font/Arial.ttf", 20)
        draw.text((10, 10),"Predicted - " +class_name, fill="red", font=font)
        draw.text((10, 60), "Actual - " + actual_category, fill="red", font=font)
        result_path = os.path.join(output_path, os.path.basename(image_path))

        result_path = result_path.replace('.jpg', '_annotated.jpg')
        image.save(result_path)
        print(f"Annotated image saved at {result_path}")

@task_wrapper
def infer_task(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule
):
    log.info("Starting testing!")
    print(f"checkpoint path file {cfg.infer.checkpoint_path_file}")
    optimized_checkpoint_filename = ""
    with open(cfg.eval.checkpoint_path_file, 'r') as file:
        optimized_checkpoint_filename = file.readline().strip()

    if optimized_checkpoint_filename:
        log.info(
            f"Loading best checkpoint: {optimized_checkpoint_filename}"
        )
        output = trainer.predict(
            model, datamodule, ckpt_path=optimized_checkpoint_filename
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        output = trainer.predict(model, datamodule, ckpt_path=optimized_checkpoint_filename)

    # Set the path for infer_images
    infer_images_dir = Path("infer_images")
    os.makedirs(infer_images_dir, exist_ok=True)
    # Delete all files in infer_images/* if any exist
    if infer_images_dir.exists() and infer_images_dir.is_dir():
        for file in infer_images_dir.glob("*"):
            file.unlink()  # Delete the file

    annotate_images(output, cfg.data.classes, "./infer_images")


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def infer(cfg: DictConfig):
    log_dir = Path(cfg.paths.log_dir)
    
    setup_logger(log_dir/"infer_log.log")

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

    if cfg.get("infer"):
        infer_task(cfg, trainer, model, datamodule)

if __name__ == "__main__":
    infer()