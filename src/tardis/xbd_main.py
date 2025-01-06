# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""xBD main training package."""

import json
import os

import hydra
import kornia
import kornia.augmentation as K
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torchgeo.trainers import SemanticSegmentationTask

from xbd_datamodule import XView2DataModuleOOD
from eurosat_xbd_utils import normalize

@hydra.main(
    version_base=None,
    config_path="./configs/xview/",
    config_name="xview_config_differentdisaster_close",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_root_dir = os.path.join(cfg.logging.path, cfg.exp_name)

    root = cfg.paths.root
    cfg.id_ood_disaster

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=default_root_dir, save_last=False
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=cfg.training.patience
    )

    tb_logger = TensorBoardLogger(save_dir=default_root_dir, name=cfg.exp_name)

    train_aug = K.AugmentationSequential(
        K.RandomRotation(degrees=90, p=0.5),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        kornia.contrib.Lambda(normalize),
        K.RandomCrop(
            (cfg.training.random_crop, cfg.training.random_crop),
            p=1.0,
            keepdim=True,
            same_on_batch=False,
        ),
        data_keys=None,
    ).to(device)

    val_aug = K.AugmentationSequential(
        kornia.contrib.Lambda(normalize),
        K.RandomCrop(
            (cfg.training.random_crop, cfg.training.random_crop),
            p=1.0,
            keepdim=True,
            same_on_batch=False,
        ),
        data_keys=None,
    ).to(device)

    test_aug = K.AugmentationSequential(
        kornia.contrib.Lambda(normalize),
        K.RandomCrop(
            (cfg.training.random_crop, cfg.training.random_crop),
            p=1.0,
            keepdim=True,
            same_on_batch=False,
        ),
        data_keys=None,
    ).to(device)

    # Initialize the data module
    datamodule = XView2DataModuleOOD(
        root=root,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        val_split_pct=cfg.training.val_split_pct,
        id_ood_disaster=cfg.id_ood_disaster,
        persistent_workers=cfg.training.persistent_workers,
        pin_memory=cfg.training.pin_memory,
        train_aug=train_aug,
        val_aug=val_aug,
        test_aug=test_aug,
    )

    task = SemanticSegmentationTask(
        model=cfg.model.type,
        backbone=cfg.model.backbone,
        weights=cfg.model.weights,
        in_channels=cfg.model.in_channels,
        num_classes=2,
        patience=cfg.model.task_patience,
        freeze_backbone=cfg.model.freeze_backbone,
        freeze_decoder=cfg.model.freeze_decoder,
        lr=cfg.model.lr,
        class_weights=None,
        ignore_index=None,
    )

    # Trainer setup
    trainer = Trainer(
        accelerator=device,
        devices=cfg.training.devices,
        default_root_dir=default_root_dir,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        num_sanity_val_steps=0,
    )

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
