# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""EuroSAT dataset."""

import json
import os

import kornia.augmentation as K
import rasterio
import torch
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.eurosat import SPATIAL_MEAN, SPATIAL_STD
from torchgeo.datasets import EuroSAT, EuroSATSpatial
from torchgeo.transforms import AugmentationSequential


class EuroSATSpatialDataModuleAug(NonGeoDataModule):
    def __init__(self, root, batch_size=64, num_workers=0, device="cuda", **kwargs):
        super().__init__(
            EuroSATSpatial, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.device = device
        self.root = root
        self.sampler = self.kwargs.pop("sampler", None)
        bands = self.kwargs.get("bands", EuroSAT.all_band_names)

        print(bands)

        self.train_aug = AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90, keepdim=True, same_on_batch=False),
            K.RandomHorizontalFlip(p=0.5, keepdim=True, same_on_batch=False),
            K.RandomVerticalFlip(p=0.5, keepdim=True, same_on_batch=False),
            K.Normalize(
                mean=list(SPATIAL_MEAN.values()),
                std=list(SPATIAL_STD.values()),
                p=1.0,
                keepdim=True,
            ),
            K.RandomSharpness(p=0.5, keepdim=True, same_on_batch=False),
            data_keys=["image"],
            same_on_batch=False,
        ).to(device)

        self.val_aug = AugmentationSequential(
            K.Normalize(
                mean=list(SPATIAL_MEAN.values()),
                std=list(SPATIAL_STD.values()),
                p=1.0,
                keepdim=True,
            ),
            data_keys=["image"],
            same_on_batch=True,
        ).to(device)

        self.test_aug = AugmentationSequential(
            K.Normalize(
                mean=list(SPATIAL_MEAN.values()),
                std=list(SPATIAL_STD.values()),
                p=1.0,
                keepdim=True,
            ),
            data_keys=["image"],
            same_on_batch=True,
        ).to(device)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = EuroSATSpatial(self.root, split="train", **self.kwargs)
            self.val_dataset = EuroSATSpatial(self.root, split="val", **self.kwargs)
        if stage == "test" or stage is None:
            self.test_dataset = EuroSATSpatial(self.root, split="test", **self.kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class EuroSATClassHoldOut(EuroSAT):
    def __init__(
        self,
        class_name,
        root,
        split="train",
        download=False,
        checksum=False,
        exclude_class=None,
        **kwargs,
    ):
        self.split = split
        self.exclude_class = exclude_class
        super().__init__(
            root=root, split=split, download=download, checksum=checksum, **kwargs
        )
        # Base URL and class name setup
        self.base_url = (
            "https://huggingface.co/datasets/brkekm/EuroSATHoldOutSplits/resolve/main/"
        )
        self._class_name = class_name.lower()

        # Override the split URLs
        self.split_urls = {
            "train": f"{self.base_url}train_{self._class_name}.txt",
            "val": f"{self.base_url}val_{self._class_name}.txt",
            "test": f"{self.base_url}test_{self._class_name}.txt",
        }

        super().__init__(
            root=root, split=split, download=download, checksum=checksum, **kwargs
        )
        self.urls = self.split_urls

    def find_classes(self, directory: str):
        """
        Find the class folders in the dataset directory, optionally including only a specific class
        for testing or excluding it for training/validation.
        """
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if self.split == "test" and self.exclude_class:
            # During testing, include only the exclude_class with index X
            classes = [
                cls for cls in classes if cls.lower() == self.exclude_class.lower()
            ]
            if not classes:
                raise FileNotFoundError(
                    f"The class {self.exclude_class} was not found in {directory} for testing."
                )
            class_to_idx = {
                self.exclude_class: -1
            }  # Assign index X to the exclude_class
        elif self.split != "test" and self.exclude_class:
            # Exclude the exclude_class during training and validation
            classes = [
                cls for cls in classes if cls.lower() != self.exclude_class.lower()
            ]
            if not classes:
                raise FileNotFoundError(
                    f"No valid class folders found in {directory} after excluding {self.exclude_class}."
                )
            class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
        else:
            # If no exclude_class is specified or not filtering, return all classes
            if not classes:
                raise FileNotFoundError(f"No valid class folders found in {directory}.")
            class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
        print("classes and class to idx", classes, class_to_idx)
        return classes, class_to_idx

    @property
    def active_classes(self):
        """Returns the list of currently active classes."""
        print("Class to idx", self.class_to_idx)
        return self.class_to_idx


class EuroSATClassHoldOutAug(NonGeoDataModule):
    def __init__(
        self, class_name, root, batch_size=64, num_workers=0, device="cuda", **kwargs
    ):
        super().__init__(
            EuroSATClassHoldOut,
            class_name=class_name,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.device = device
        self.root = root
        self.class_name = class_name
        kwargs.get("bands", EuroSAT.all_band_names)
        MEAN, STD = self.load_statistics(os.path.dirname(self.root), class_name)
        print(f"Loaded statistics for {class_name}: MEAN={MEAN}, STD={STD}")

        self.train_aug = AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90, keepdim=True, same_on_batch=False),
            K.RandomHorizontalFlip(p=0.5, keepdim=True, same_on_batch=False),
            K.RandomVerticalFlip(p=0.5, keepdim=True, same_on_batch=False),
            K.Normalize(
                mean=list(MEAN.values()),
                std=list(STD.values()),
                p=1.0,
                keepdim=True,
            ),
            K.RandomSharpness(p=0.5, keepdim=True, same_on_batch=False),
            data_keys=["image", "label"],
            same_on_batch=False,
        )

        self.val_aug = AugmentationSequential(
            K.Normalize(
                mean=list(MEAN.values()),
                std=list(STD.values()),
                p=1.0,
                keepdim=True,
            ),
            data_keys=["image", "label"],
            same_on_batch=True,
        )

        self.test_aug = AugmentationSequential(
            K.Normalize(
                mean=list(MEAN.values()),
                std=list(STD.values()),
                p=1.0,
                keepdim=True,
            ),
            data_keys=["image", "label"],
            same_on_batch=True,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = EuroSATClassHoldOut(split="train", **self.kwargs)
            self.val_dataset = EuroSATClassHoldOut(split="val", **self.kwargs)
        if stage == "test" or stage is None:
            self.test_dataset = EuroSATClassHoldOut(split="test", **self.kwargs)

    def load_statistics(self, root, class_name):
        statistics_file_path = os.path.join(root, "eurosat_holdout_statistics.json")
        with open(statistics_file_path, "r") as f:
            all_statistics = json.load(f)
        mean_key = f"{class_name}_MEAN"
        std_key = f"{class_name}_STD"
        return all_statistics.get(mean_key, {}), all_statistics.get(std_key, {})


class EuroSATClassHoldOutGeoExtract(EuroSAT):
    """
    Dataset class for EuroSAT holdout experiment inference with geolocation extraction.
    """

    def __init__(
        self,
        class_name,
        root,
        split_dir,
        split="train",
        download=False,
        checksum=False,
        exclude_class=None,
        transforms=None,
        **kwargs,
    ):
        self.split = split
        self.exclude_class = exclude_class

        # Base URL and class name setup
        self.base_url = (
            "https://huggingface.co/datasets/brkekm/EuroSATHoldOutSplits/resolve/main/"
        )
        self._class_name = class_name.lower()

        # Override the split URLs
        self.split_urls = {
            "train": f"{self.base_url}train_{self._class_name}.txt",
            "val": f"{self.base_url}val_{self._class_name}.txt",
            "test": f"{self.base_url}test_{self._class_name}.txt",
        }

        super().__init__(
            root=root, split=split, download=download, checksum=checksum, **kwargs
        )
        self.urls = self.split_urls
        self.transforms = transforms
        print("self.transforms", self.transforms)

    def find_classes(self, directory: str):
        """
        Find the class folders in the dataset directory, optionally including only a specific class
        for testing or excluding it for training/validation.
        """
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if self.split == "test" and self.exclude_class:
            # During testing, include only the exclude_class with index X
            classes = [
                cls for cls in classes if cls.lower() == self.exclude_class.lower()
            ]
            if not classes:
                raise FileNotFoundError(
                    f"The class {self.exclude_class} was not found in {directory} for testing."
                )
            class_to_idx = {
                self.exclude_class: -1
            }  # Assign index X to the exclude_class
        elif self.split != "test" and self.exclude_class:
            # Exclude the exclude_class during training and validation
            classes = [
                cls for cls in classes if cls.lower() != self.exclude_class.lower()
            ]
            if not classes:
                raise FileNotFoundError(
                    f"No valid class folders found in {directory} after excluding {self.exclude_class}."
                )
            class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
        else:
            # If no exclude_class is specified or not filtering, return all classes
            if not classes:
                raise FileNotFoundError(f"No valid class folders found in {directory}.")
            class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
        print("classes and class to idx", classes, class_to_idx)
        return classes, class_to_idx

    @property
    def active_classes(self):
        """Returns the list of currently active classes."""
        print("Class to IDX", self.class_to_idx)
        return self.class_to_idx

    def _load_split_data(self, split_dir):
        split_file = f"{split_dir}/eurosat-{self.split}.txt"
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"No such file for split data: {split_file}")

        with open(split_file, "r") as file:
            for line in file:
                filename = line.strip()
                class_label = filename.split("_")[
                    0
                ]  # Assuming class_label is the prefix
                full_path = os.path.join(self.root_dir, class_label, filename)
                if os.path.exists(
                    full_path.replace(".jpg", ".tif")
                ):  # Ensure TIFF file exists
                    self.samples.append(
                        (full_path.replace(".jpg", ".tif"), class_label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_class = self.samples[idx]
        image = self._load_image(path_class[0])
        image = self.transforms(image)
        return {"image": image, "label": path_class[1], "path": path_class[0]}

    def _load_image(self, path):
        with rasterio.open(path) as src:
            array = src.read().astype("float32")
        tensor = torch.from_numpy(array)
        return tensor
