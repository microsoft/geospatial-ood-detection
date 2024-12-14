# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""EuroSAT main training package."""

import json
import os

import hydra
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint,
                                         TQDMProgressBar)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torchgeo.trainers import ClassificationTask

from .eurosat_datamodule import EuroSATSpatialDataModuleAug


@hydra.main(
    version_base=None,
    config_path="./configs/eurosat/",
    config_name="eurosat_spatial_config.yaml",
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    default_root_dir = os.path.join(cfg.logging.path, cfg.exp_name)

    # Initialize the classification task
    task = ClassificationTask(
        model=cfg.model.name,
        weights=cfg.model.weights,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        loss=cfg.model.loss,
        lr=cfg.model.lr,
        patience=cfg.model.patience,
        freeze_backbone=False,
    )

    # Setup callbacks for model checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=default_root_dir, save_last=False
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=cfg.model.patience
    )

    tb_logger = TensorBoardLogger(save_dir=default_root_dir, name=cfg.exp_name)

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stopping_callback, TQDMProgressBar()],
        fast_dev_run=cfg.trainer.fast_dev_run,
        logger=tb_logger,
        default_root_dir=default_root_dir,
        min_epochs=cfg.trainer.min_epochs,
        max_epochs=cfg.trainer.max_epochs,
    )

    # Spatial split datamodule
    datamodule = EuroSATSpatialDataModuleAug(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        sampler=None,
    )

    # Fit and test the model
    trainer.fit(model=task, datamodule=datamodule)
    test_results = trainer.test(model=task, datamodule=datamodule)

    results = {
        "test_results": test_results,
        "best_model_path": checkpoint_callback.best_model_path,
    }

    results_dir = default_root_dir
    os.makedirs(results_dir, exist_ok=True)

    with open(
        os.path.join(results_dir, f"test_eval_results_{cfg.exp_name}.json"), "w"
    ) as f:
        json.dump(results, f, indent=4)

    print(
        f"Results saved to test_eval_results_{cfg.exp_name}.json in the experiment directory."
    )


if __name__ == "__main__":
    main()
