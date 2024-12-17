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
from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import (
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
)
from torchgeo.datasets.xview import XView2

# class XView2DistShift(XView2):
#     """
#     A subclass of the XView2 dataset designed to reformat the original train/test splits
#     based on specific in-domain (ID) and out-of-domain (OOD) disasters.

#     This class allows for the selection of particular disasters to be used as the
#     training set (in-domain) and test set (out-of-domain). The dataset can be split
#     according to the disaster names specified by the user, enabling the model to train
#     on one disaster type and evaluate on a different, out-of-domain disaster. The goal
#     is to test the generalization ability of models trained on one disaster to perform
#     on others.
#     """

#     classes = ["background", "building"]

#     # List of possible disaster names
#     valid_disasters = [
#         'hurricane-harvey', 'socal-fire', 'hurricane-matthew', 'mexico-earthquake',
#         'guatemala-volcano', 'santa-rosa-wildfire', 'palu-tsunami', 'hurricane-florence',
#         'hurricane-michael', 'midwest-flooding'
#     ]

#     def __init__(
#         self,
#         root: Path = "data",
#         split: str = "train",
#         id_ood_disaster: List[Dict[str, str]] = [{"disaster_name": "hurricane-matthew", "pre-post": "post"}, {"disaster_name": "mexico-earthquake", "pre-post": "post"}],
#         transforms: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
#         checksum: bool = False,
#         **kwargs
#     ) -> None:
#         """Initialize the XView2DistShift dataset instance.

#         Args:
#             root: Root directory where the dataset is located.
#             split: One of "train" or "test".
#             id_ood_disaster: List containing in-distribution and out-of-distribution disaster names.
#             transforms: a function/transform that takes input sample and its target as
#                 entry and returns a transformed version
#             checksum: if True, check the MD5 of the downloaded files (may be slow)

#         Raises:
#             AssertionError: If *split* is invalid.
#             ValueError: If a disaster name in `id_ood_disaster` is not one of the valid disasters.
#             DatasetNotFoundError: If dataset is not found.
#         """
#         assert split in ["train", "test"], "Split must be either 'train' or 'test'."
#         # Validate that the disasters are valid

#         print("id_ood_disaster", id_ood_disaster)
#         if id_ood_disaster[0]['disaster_name'] not in self.valid_disasters or id_ood_disaster[1]['disaster_name'] not in self.valid_disasters:
#             raise ValueError(f"Invalid disaster names. Valid options are: {', '.join(self.valid_disasters)}")

#         self.root = root
#         self.split = split
#         self.transforms = transforms
#         self.checksum = checksum

#         self._verify()

#         # Load all files and compute basenames and disasters only once
#         self.all_files = self._initialize_files(root)

#         # Split logic by disaster and pre-post type
#         self.files = self._load_split_files_by_disaster_and_type(self.all_files, id_ood_disaster[0], id_ood_disaster[1])
#         print(f"Loaded for disasters ID and OOD: {len(self.files['train'])} train, {len(self.files['test'])} test files.")

#     def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
#         """Get an item from the dataset at the given index."""
#         file_info = (
#             self.files["train"][index]
#             if self.split == "train"
#             else self.files["test"][index])
#         image = self._load_image(file_info["image"]).float().to("cuda")
#         mask = self._load_target(file_info["mask"]).long().to("cuda")
#         mask[mask == 2] = 1
#         mask[(mask == 3) | (mask == 4)] = 0

#         sample = {"image": image, "mask": mask, "file_info":file_info}

#         if self.transforms:
#             sample = self.transforms(sample)

#         return sample

#     def __len__(self) -> int:
#         """Return the total number of samples in the dataset."""
#         return (
#             len(self.files["train"])
#             if self.split == "train"
#             else len(self.files["test"])
#         )

#     def _initialize_files(self, root: Path) -> List[Dict[str, str]]:
#         """Initialize the dataset by loading file paths and computing basenames with sample numbers."""
#         all_files = []
#         for split in self.metadata.keys():
#             image_root = os.path.join(root, split, "images")
#             mask_root = os.path.join(root, split, "targets")
#             images = glob.glob(os.path.join(image_root, "*.png"))

#             # Extract basenames while preserving the event-name and sample number
#             for img in images:
#                 basename_parts = os.path.basename(img).split("_")
#                 event_name = basename_parts[0]  # e.g., mexico-earthquake
#                 sample_number = basename_parts[1]  # e.g., 00000001
#                 basename = (
#                     f"{event_name}_{sample_number}"  # e.g., mexico-earthquake_00000001
#                 )


#                 file_info = {
#                     "image": img,
#                     "mask": os.path.join(
#                         mask_root, f"{basename}_pre_disaster_target.png"
#                     ),
#                     "basename": basename,
#                 }
#                 all_files.append(file_info)
#         return all_files

#     def _load_split_files_by_disaster_and_type(
#         self, files: List[Dict[str, str]], id_disaster: Dict[str, str], ood_disaster: Dict[str, str]
#     ) -> Dict[str, List[Dict[str, str]]]:
#         """
#         Return the paths of the files for the train (ID) and test (OOD) sets based on the specified disaster name
#         and pre-post disaster type.

#         Args:
#             files: List of file paths with their corresponding information.
#             id_disaster: Dictionary specifying in-domain (ID) disaster and type (e.g., {"disaster_name": "guatemala-volcano", "pre-post": "pre"}).
#             ood_disaster: Dictionary specifying out-of-domain (OOD) disaster and type (e.g., {"disaster_name": "mexico-earthquake", "pre-post": "post"}).

#         Returns:
#             A dictionary containing 'train' (ID) and 'test' (OOD) file lists.
#         """
#         train_files = []
#         test_files = []
#         disaster_list = []

#         for file_info in files:
#             basename = file_info["basename"]
#             disaster_name = basename.split("_")[0]  # Extract disaster name from basename
#             pre_post = ("pre" if "pre_disaster" in file_info["image"] else "post")  # Identify pre/post type

#             disaster_list.append(disaster_name)

#             # Filter for in-domain (ID) training set
#             if disaster_name == id_disaster["disaster_name"]:
#                 if id_disaster.get("pre-post") == "both" or id_disaster["pre-post"] == pre_post:
#                     image = (
#                         file_info["image"].replace("post_disaster", "pre_disaster")
#                         if pre_post == "pre"
#                         else file_info["image"]
#                     )
#                     mask = (
#                         file_info["mask"].replace("post_disaster", "pre_disaster")
#                         if pre_post == "pre"
#                         else file_info["mask"]
#                     )
#                     train_files.append(dict(image=image, mask=mask))

#             # Filter for out-of-domain (OOD) test set
#             if disaster_name == ood_disaster["disaster_name"]:
#                 if ood_disaster.get("pre-post") == "both" or ood_disaster["pre-post"] == pre_post:
#                     test_files.append(file_info)

#         return {"train": train_files, "test": test_files, "disasters":disaster_list}


class XView2DistShift(XView2):
    """
    A subclass of the XView2 dataset designed to reformat the original train/test splits
    based on specific in-domain (ID) and out-of-domain (OOD) disasters.

    This class allows for the selection of particular disasters to be used as the
    training set (in-domain) and test set (out-of-domain). The dataset can be split
    according to the disaster names specified by the user, enabling the model to train
    on one disaster type and evaluate on a different, out-of-domain disaster. The goal
    is to test the generalization ability of models trained on one disaster to perform
    on others.
    """

    classes = ["background", "building"]

    valid_disasters = [
        "tuscaloosa-tornado",
        "palu-tsunami",
        "socal-fire",
        "hurricane-matthew",
        "woolsey-fire",
        "hurricane-florence",
        "pinery-bushfire",
        "portugal-wildfire",
        "hurricane-harvey",
        "hurricane-michael",
        "santa-rosa-wildfire",
        "guatemala-volcano",
        "lower-puna-volcano",
        "sunda-tsunami",
        "midwest-flooding",
        "moore-tornado",
        "joplin-tornado",
        "nepal-flooding",
        "mexico-earthquake",
    ]

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        id_ood_disaster: List[Dict[str, str]] = [
            {"disaster_name": "hurricane-matthew", "pre-post": "post"},
            {"disaster_name": "mexico-earthquake", "pre-post": "post"},
        ],
        transforms: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        download: bool = False,
        checksum: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the XView2DistShift dataset instance.

        Args:
            root: Root directory where the dataset is located.
            split: One of "train" or "test".
            id_ood_disaster: List containing in-distribution and out-of-distribution disaster names.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If *split* is invalid.
            ValueError: If a disaster name in `id_ood_disaster` is not one of the valid disasters.
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in ["train", "test"], "Split must be either 'train' or 'test'."
        # Validate that the disasters are valid

        if (
            id_ood_disaster[0]["disaster_name"] not in self.valid_disasters
            or id_ood_disaster[1]["disaster_name"] not in self.valid_disasters
        ):
            raise ValueError(
                f"Invalid disaster names. Valid options are: {', '.join(self.valid_disasters)}"
            )

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        # Load all files and compute basenames and disasters only once
        self.all_files = self._initialize_files(root)

        # Split logic by disaster and pre-post type
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

        image = self._load_image(file_info["image"]).float().to("cuda")
        mask = self._load_target(file_info["mask"]).long().to("cuda")
        mask[mask == 2] = 1
        mask[(mask == 3) | (mask == 4)] = 0

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
        for split in list(self.metadata.keys()) + ["hold"]:
            image_root = os.path.join(root, split, "images")
            mask_root = os.path.join(root, split, "targets")
            images = glob.glob(os.path.join(image_root, "*.png"))

            # Extract basenames while preserving the event-name and sample number
            for img in images:
                basename_parts = os.path.basename(img).split("_")
                event_name = basename_parts[0]  # e.g., mexico-earthquake
                sample_number = basename_parts[1]  # e.g., 00000001
                basename = (
                    f"{event_name}_{sample_number}"  # e.g., mexico-earthquake_00000001
                )

                file_info = {
                    "image": img,
                    "mask": os.path.join(
                        mask_root, f"{basename}_pre_disaster_target.png"
                    ),
                    "basename": basename,
                }
                all_files.append(file_info)
        return all_files

    def _load_split_files_by_disaster_and_type(
        self,
        files: List[Dict[str, str]],
        id_disaster: Dict[str, str],
        ood_disaster: Dict[str, str],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Return the paths of the files for the train (ID) and test (OOD) sets based on the specified disaster name
        and pre-post disaster type.

        Args:
            files: List of file paths with their corresponding information.
            id_disaster: Dictionary specifying in-domain (ID) disaster and type (e.g., {"disaster_name": "guatemala-volcano", "pre-post": "pre"}).
            ood_disaster: Dictionary specifying out-of-domain (OOD) disaster and type (e.g., {"disaster_name": "mexico-earthquake", "pre-post": "post"}).

        Returns:
            A dictionary containing 'train' (ID) and 'test' (OOD) file lists.
        """
        train_files = []
        test_files = []
        disaster_list = []

        for file_info in files:
            basename = file_info["basename"]
            disaster_name = basename.split("_")[
                0
            ]  # Extract disaster name from basename
            pre_post = (
                "pre" if "pre_disaster" in file_info["image"] else "post"
            )  # Identify pre/post type

            disaster_list.append(disaster_name)

            # Filter for in-domain (ID) training set
            if disaster_name == id_disaster["disaster_name"]:
                if (
                    id_disaster.get("pre-post") == "both"
                    or id_disaster["pre-post"] == pre_post
                ):
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
            if disaster_name == ood_disaster["disaster_name"]:
                if (
                    ood_disaster.get("pre-post") == "both"
                    or ood_disaster["pre-post"] == pre_post
                ):
                    test_files.append(file_info)

        return {"train": train_files, "test": test_files, "disasters": disaster_list}

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        exists = []
        for split_info in list(self.metadata.values()) + [
            {"directory": "hold", "filename": "hold_images_labels_targets.tar.gz"}
        ]:
            # print(split_info)
            for directory in ["images", "targets"]:
                # print("os.path.join(self.root, split_info['directory'], directory)", os.path.join(self.root, split_info['directory'], directory))
                exists.append(
                    os.path.exists(
                        os.path.join(self.root, split_info["directory"], directory)
                    )
                )
                # print("exists1", exists)
        if all(exists):
            return

        # Check if .tar.gz files already exists (if so then extract)
        exists = []
        for split_info in list(self.metadata.values()) + [
            {"directory": "hold", "filename": "hold_images_labels_targets.tar.gz"}
        ]:
            filepath = os.path.join(self.root, split_info["filename"])
            print(filepath)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, split_info["md5"]):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)
        # print("exists2", exists)

        if all(exists):
            return

        raise DatasetNotFoundError(self)


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

        print("Root", root)

    def setup(self, stage=None):
        """
        Setup method that initializes datasets for the given stage.
        Can be used to split into training, validation, and test sets.
        """
        if stage == "fit" or stage is None:
            # Prepare training and validation datasets
            full_train_dataset = XView2DistShift(
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
            self.test_dataset = XView2DistShift(
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
            shuffle=False,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
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
            drop_last=True,
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
            drop_last=True,
        )
