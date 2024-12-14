# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""xBD dataset."""

import glob
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Callable, Dict, List

import rasterio
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class XView2_OOD(Dataset):
    """xView2 dataset for building disaster damage segmentation."""

    classes = ["background", "building"]

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        id_ood_disaster: List[str] = None,
        transforms: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Initialize the xView2 dataset instance.

        Args:
            root: Root directory where the dataset is located.
            split: One of "train" or "test".
            id_ood_disaster: List containing in-distribution and out-of-distribution disaster names.
            transforms: Optional transforms to be applied on a sample.
        """
        assert split in ["train", "test"], "Split must be either 'train' or 'test'."
        self.root = root
        self.split = split
        self.transforms = transforms

        # Load all files and compute basenames and disasters only once
        self.tiers = ["tier1", "tier3"]
        print("Initializing dataset with tiers:", self.tiers)
        self.all_files = self._initialize_files(root)
        print("Initialized:", len(self.all_files), "files")

        # Split logic by disaster and pre-post type
        if id_ood_disaster is not None:
            self.files = self._load_split_files_by_disaster_and_type(
                self.all_files, id_ood_disaster[0], id_ood_disaster[1]
            )
            print(
                f"Loaded for disasters ID and OOD: {len(self.files['train'])} train, {len(self.files['test'])} test files."
            )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset at the given index."""
        file_info = (
            self.files["train"][index]
            if self.split == "train"
            else self.files["test"][index]
        )

        image = self._load_image(file_info["image"]).to("cuda")
        mask = self._load_mask(file_info["mask"]).long().to("cuda")

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return (
            len(self.files["train"])
            if self.split == "train"
            else len(self.files["test"])
        )

    def _initialize_files(self, root: Path) -> List[Dict[str, str]]:
        """Initialize the dataset by loading file paths and computing basenames with sample numbers."""
        all_files = []
        for tier in self.tiers:
            image_root = os.path.join(root, tier, "images")
            mask_root = os.path.join(root, tier, "targets")

            images = glob.glob(os.path.join(image_root, "*.tif"))

            # Extract basenames while preserving the event-name and sample number
            for img in images:
                basename_parts = os.path.basename(img).split("_")
                event_name = basename_parts[0]  # e.g., guatemala-volcano
                sample_number = basename_parts[1]  # e.g., 00000001
                basename = (
                    f"{event_name}_{sample_number}"  # e.g., guatemala-volcano_00000001
                )

                file_info = {
                    "image": img,
                    "mask": os.path.join(
                        mask_root, f"{basename}_pre_disaster_target.tif"
                    ),
                    "basename": basename,
                    "tier": tier,
                }
                all_files.append(file_info)
        return all_files

    def _load_split_files_by_disaster_and_type(
        self, files: List[Dict[str, str]], id: Dict[str, str], ood: Dict[str, str]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Return the paths of the files for the train (ID) and test (OOD) sets based on the specified disaster name
        and pre-post disaster type.

        Args:
            files: List of file paths with their corresponding information.
            id: Dictionary specifying in-domain (ID) disaster and type (e.g., {"disaster_name": "guatemala-volcano", "pre-post": "pre"}).
            ood: Dictionary specifying out-of-domain (OOD) disaster and type (e.g., {"disaster_name": "mexico-earthquake", "pre-post": "post"}).

        Returns:
            A dictionary containing 'train' (ID) and 'test' (OOD) file lists.
        """
        train_files = []
        test_files = []

        for file_info in files:
            basename = file_info["basename"]
            disaster_name = basename.split("_")[
                0
            ]  # Extract disaster name from basename
            pre_post = (
                "pre" if "pre_disaster" in file_info["image"] else "post"
            )  # Identify pre/post type

            # Filter for in-domain (ID) training set
            if disaster_name == id["disaster_name"]:
                if id.get("pre-post") == "both" or id["pre-post"] == pre_post:
                    image = (
                        file_info["image"].replace("post_disaster", "pre_disaster")
                        if pre_post == "pre"
                        else file_info["image"]
                    )
                    mask = (
                        file_info["mask"].replace("post_disaster", "pre_disaster")
                        if pre_post == "pre"
                        else file_info["mask"]
                    )
                    train_files.append(dict(image=image, mask=mask))

            # Filter for out-of-domain (OOD) test set
            if disaster_name == ood["disaster_name"]:
                if ood.get("pre-post") == "both" or ood["pre-post"] == pre_post:
                    test_files.append(file_info)

        return {"train": train_files, "test": test_files}

    def _load_image(self, path: str) -> torch.Tensor:
        """Load the image and convert to a floating point tensor."""
        with rasterio.open(path) as f:
            array = f.read()
            tensor = torch.from_numpy(array).float()
            tensor = tensor.squeeze()
            return tensor

    def _load_mask(self, path: str) -> torch.Tensor:
        """Load the image, convert to a floating point tensor, and set all values greater than 0 to 1."""
        with rasterio.open(path) as f:
            array = f.read()
            tensor = torch.from_numpy(array).float()
            tensor = tensor.squeeze()
            tensor[tensor == 2] = 1
            tensor[(tensor == 3) | (tensor == 4)] = 0
            return tensor

    def count_total_pairs(self) -> int:
        """Count the total number of image-mask pairs in the dataset."""
        return len(self.files)

    def count_images_per_disaster(self) -> Dict[str, int]:
        """Count how many image-mask pairs are present for each disaster."""
        disaster_count = defaultdict(int)

        for file_info in self.files:
            # Extract the disaster name from the image path (before the first underscore)
            image_path = file_info["image"]
            disaster_name = os.path.basename(image_path).split("_")[0]
            disaster_count[disaster_name] += 1
        return dict(disaster_count)

    def count_pre_post_disaster_images(self) -> Dict[str, int]:
        """Count the number of pre-disaster and post-disaster images."""
        counts = {"pre": 0, "post": 0}

        for file_info in self.files:
            image_path = file_info["image"]
            if "pre_disaster" in image_path:
                counts["pre"] += 1
            elif "post_disaster" in image_path:
                counts["post"] += 1
        return counts


class XView2DataModuleOOD(LightningDataModule):
    def __init__(
        self,
        root,
        batch_size,
        num_workers,
        val_split_pct,
        id_ood_disaster,
        persistent_workers=False,
        pin_memory=False,
        train_aug=None,
        val_aug=None,
        test_aug=None,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.id_ood_disaster = id_ood_disaster
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        self.train_aug = train_aug
        self.val_aug = val_aug
        self.test_aug = test_aug

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Setup method that initializes datasets for the given stage.
        Can be used to split into training, validation, and test sets.
        """
        if stage == "fit" or stage is None:
            # Prepare training and validation datasets
            full_train_dataset = XView2_OOD(
                root=self.root,
                split="train",
                id_ood_disaster=self.id_ood_disaster,
                transforms=self.train_aug,  # Apply train augmentations
            )

            # Split the full training dataset into train and validation sets
            val_size = int(self.val_split_pct * len(full_train_dataset))
            train_size = len(full_train_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, [train_size, val_size]
            )

            # Apply validation augmentations
            self.val_dataset.dataset.transforms = self.val_aug

        if stage == "test" or stage is None:
            # Initialize test dataset (out-of-domain disaster)
            self.test_dataset = XView2_OOD(
                root=self.root,
                split="test",
                id_ood_disaster=self.id_ood_disaster,
                transforms=self.test_aug,  # Apply test augmentations
            )

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
