# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing utilities for processing ID and WILD sets."""

import os
from collections import defaultdict
from typing import Any

import kornia.augmentation as K
import numpy as np
import rasterio
import torch
import tqdm
from pyproj import Transformer
from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import FieldsOfTheWorld
from torchgeo.transforms import AugmentationSequential


class FTWDataModuleOOD(NonGeoDataModule):
    """LightningDataModule implementation for the FTW dataset."""

    def __init__(
        self,
        root_folder_torchgeo: str,
        batch_size: int = 64,
        num_workers: int = 0,
        train_countries: list[str] = ["france"],
        val_countries: list[str] = ["france"],
        test_countries: list[str] = ["france"],
        download: bool = False,
        sample_N_from_each_country: int = 50,
        target: str = "2-class",
        **kwargs: Any,
    ) -> None:
        """Initialize a new FTWDataModule instance.

        Note: you can pass train_batch_size, val_batch_size, test_batch_size to
            control the batch sizes of each DataLoader independently.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            train_countries: List of countries to use training splits from
            val_countries: List of countries to use validation splits from
            test_countries: List of countries to use test splits from
            **kwargs: Additional keyword arguments passed to
                :class:`~src.datasets.FTW`.
        """
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWDataModule")

        self.train_countries = train_countries
        self.val_countries = val_countries
        self.test_countries = test_countries
        self.download = download
        self.sample_N_from_each_country = sample_N_from_each_country
        self.target = target
        self.root_folder_torchgeo = root_folder_torchgeo

        print("Loaded datamodule with:")
        print(f"Train countries: {self.train_countries}")
        print(f"Val countries: {self.val_countries}")
        print(f"Test countries: {self.test_countries}")

        self.train_aug = None

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            data_keys=["image", "mask"],
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )
        super().__init__(FieldsOfTheWorld, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = FieldsOfTheWorld(
                self.root_folder_torchgeo,
                countries=self.train_countries,
                target=self.target,
                split="train",
                download=self.download,
                sample_N=self.sample_N_from_each_country,
            )
            print(
                f"Train dataset: {len(self.train_dataset)} for the countries {self.train_countries}"
            )

        if stage in ["fit", "validate"]:
            self.val_dataset = FieldsOfTheWorld(
                self.root_folder_torchgeo,
                countries=self.val_countries,
                target=self.target,
                split="val",
                download=self.download,
                sample_N=self.sample_N_from_each_country,
            )
            print(
                f"Val dataset: {len(self.val_dataset)} for the countries {self.val_countries}"
            )

        if stage == "test":
            self.test_dataset = FieldsOfTheWorld(
                self.root_folder_torchgeo,
                countries=self.test_countries,
                target=self.target,
                split="test",
                download=self.download,
                sample_N=self.sample_N_from_each_country,
            )
            print(
                f"Test dataset: {len(self.test_dataset)} for the countries {self.test_countries}"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )


class WILDGeoTIFFDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.paired_files = self._group_files_by_randomid()
        self.valid_pairs, self.coords = self._filter_valid_pairs()

    def _group_files_by_randomid(self):
        """Group files by the randomid (first part of the filename)."""
        grouped_files = defaultdict(list)
        for f in os.listdir(self.directory):
            if f.endswith(".tif"):
                randomid = f.split("_")[0]
                grouped_files[randomid].append(f)

        # Ensure pairs and sort so that "plant" comes before "harvest"
        paired_files = []
        for randomid, files in grouped_files.items():
            if len(files) == 2:  # Expect exactly two files per randomid
                files.sort(key=lambda x: "plant" in x)  # Sort "plant" first
                paired_files.append(files)
        return paired_files

    def _compute_coords(self, filepath):
        """Compute center coordinates of the image."""
        _, bounds, crs = self._load_image(filepath, return_meta=True)
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2

        # Transform coordinates to WGS84 (latitude and longitude)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = transformer.transform(center_x, center_y)
        return torch.tensor([center_lat, center_lon], dtype=torch.float)

    def _filter_valid_pairs(self):
        valid_pairs = []
        coords_list = []
        for pair in self.paired_files:
            coords = [
                self._compute_coords(os.path.join(self.directory, filename))
                for filename in pair
            ]

            if torch.equal(coords[0], coords[1]):
                valid_pairs.append(pair)
                coords_list.append(coords)
            else:
                print(f"Warning: Coordinates differ in pair {pair}:")
                print(f" - {pair[0]} coords: {coords[0]}")
                print(f" - {pair[1]} coords: {coords[1]}")

        return valid_pairs, coords_list

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        file_pair = self.valid_pairs[idx]
        tensors = []
        pair_coords = self.coords[idx]  # Use precomputed coordinates

        for filename in file_pair:
            filepath = os.path.join(self.directory, filename)
            tensor = self._load_image(filepath)
            tensors.append(tensor)

        # Concatenate the tensors along the first dimension (channels)
        combined_tensor = torch.cat(tensors, dim=0)

        sample = {"image": combined_tensor, "coords": pair_coords}
        return sample

    def _load_image(self, path: str, return_meta: bool = False):
        """Load a single image and convert it to a tensor.
        Args:
            path: Path to the image file.
            return_meta: If True, also return the geospatial metadata.
        Returns:
            The image tensor, and optionally the metadata (bounds, CRS).
        """
        with rasterio.open(path) as dataset:
            array = dataset.read()  # Read all bands
            # Handle specific numpy dtypes
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            elif array.dtype == np.uint32:
                array = array.astype(np.int64)

            # Convert to PyTorch tensor
            tensor = torch.tensor(array).float()

            if return_meta:
                bounds = dataset.bounds
                crs = dataset.crs
                return tensor, bounds, crs

        return tensor


def process_WILD_dataloader(
    dataloader,
    ood_model,
    return_batch=False,
    return_f_pred=False,
    return_g_pred=False,
    return_thresholded_g_pred=False,
    return_coords=False,
    upsample=True,
    max_batches=None,
):
    def process_batch(batch):
        images = batch["image"].to("cuda")
        coords = np.squeeze(np.array(batch["coords"]))

        coords = coords[0]  # Take the first pairs' coordinates

        # Prediction
        with torch.no_grad():
            f_preds, g_pred_probs = ood_model.f_g_prediction(images, upsample=upsample)

        # Move predictions back to CPU
        f_preds = (
            f_preds.to("cpu")
            if isinstance(f_preds, torch.Tensor)
            else torch.tensor(f_preds)
        )
        g_pred_probs = (
            g_pred_probs.to("cpu")
            if isinstance(g_pred_probs, torch.Tensor)
            else torch.tensor(g_pred_probs)
        )

        results = {}

        if return_batch:
            results["batch"] = batch

        if return_f_pred:
            f_preds_single_channel = torch.argmax(f_preds, dim=1)
            results["f_preds"] = f_preds_single_channel.numpy()

        if return_g_pred:
            results["g_pred_probs"] = g_pred_probs.numpy()

        if return_thresholded_g_pred:
            thresholded_g_pred_probs = np.where(g_pred_probs > 0.5, 1, 0)
            results["thresholded_g_pred_probs"] = thresholded_g_pred_probs

        if return_coords:
            results["coords"] = coords

        # Clear GPU memory
        del images, f_preds, g_pred_probs
        torch.cuda.empty_cache()

        return results

    all_f_preds = []
    all_g_pred_probs = []
    all_thresholded_g_pred_probs = []
    all_coords = []
    all_batches = []

    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        if max_batches is not None and i >= max_batches:
            break
        results = process_batch(batch)

        if return_batch:
            all_batches.append(results["batch"]["image"])

        if return_f_pred:
            all_f_preds.append(results["f_preds"])

        if return_g_pred:
            all_g_pred_probs.append(results["g_pred_probs"])

        if return_thresholded_g_pred:
            all_thresholded_g_pred_probs.append(results["thresholded_g_pred_probs"])

        if return_coords:
            all_coords.append(results["coords"])

    concatenated_results = {}

    if return_batch:
        concatenated_results["batch"] = torch.cat(all_batches, dim=0).cpu()

    if return_f_pred:
        concatenated_results["f_preds"] = np.concatenate(all_f_preds, axis=0)

    if return_g_pred:
        concatenated_results["g_pred_probs"] = np.concatenate(all_g_pred_probs, axis=0)

    if return_thresholded_g_pred:
        concatenated_results["thresholded_g_pred_probs"] = np.concatenate(
            all_thresholded_g_pred_probs, axis=0
        )

    if return_coords:
        concatenated_results["coords"] = np.concatenate(all_coords, axis=0)

    return concatenated_results


def process_ID_dataloader(
    dataloader,
    ood_model,
    return_batch=False,
    return_f_pred=False,
    return_g_pred=False,
    return_thresholded_g_pred=False,
    return_coords=False,
    return_masks=False,
    upsample=True,
    max_batches=None,
):
    def process_batch(batch):
        images = batch["image"].to("cuda")
        masks = batch["mask"].to("cuda")
        coords = np.squeeze(np.array(batch["coords"]))

        # Prediction
        with torch.no_grad():
            f_preds, g_pred_probs = ood_model.f_g_prediction(images, upsample=upsample)

        # Move predictions back to CPU
        f_preds = (
            f_preds.to("cpu")
            if isinstance(f_preds, torch.Tensor)
            else torch.tensor(f_preds)
        )
        g_pred_probs = (
            g_pred_probs.to("cpu")
            if isinstance(g_pred_probs, torch.Tensor)
            else torch.tensor(g_pred_probs)
        )
        masks = (
            masks.to("cpu") if isinstance(masks, torch.Tensor) else torch.tensor(masks)
        )

        results = {}

        if return_batch:
            results["batch"] = batch

        if return_f_pred:
            f_preds_single_channel = torch.argmax(f_preds, dim=1)
            results["f_preds"] = f_preds_single_channel.numpy()

        if return_g_pred:
            results["g_pred_probs"] = g_pred_probs.numpy()

        if return_thresholded_g_pred:
            thresholded_g_pred_probs = np.where(g_pred_probs > 0.5, 1, 0)
            results["thresholded_g_pred_probs"] = thresholded_g_pred_probs

        if return_coords:
            results["coords"] = coords

        if return_masks:
            results["masks"] = masks.numpy()

        # Clear GPU memory
        del images, f_preds, g_pred_probs, masks
        torch.cuda.empty_cache()

        return results

    all_f_preds = []
    all_g_pred_probs = []
    all_thresholded_g_pred_probs = []
    all_coords = []
    all_masks = []
    all_batches = []

    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        if max_batches is not None and i >= max_batches:
            break
        results = process_batch(batch)

        if return_batch:
            all_batches.append(results["batch"]["image"])

        if return_f_pred:
            all_f_preds.append(results["f_preds"])

        if return_g_pred:
            all_g_pred_probs.append(results["g_pred_probs"])

        if return_thresholded_g_pred:
            all_thresholded_g_pred_probs.append(results["thresholded_g_pred_probs"])

        if return_coords:
            all_coords.append(results["coords"])

        if return_masks:
            all_masks.append(results["masks"])

    concatenated_results = {}

    if return_batch:
        concatenated_results["batch"] = torch.cat(all_batches, dim=0).cpu()

    if return_f_pred:
        concatenated_results["f_preds"] = np.concatenate(all_f_preds, axis=0)

    if return_g_pred:
        concatenated_results["g_pred_probs"] = np.concatenate(all_g_pred_probs, axis=0)

    if return_thresholded_g_pred:
        concatenated_results["thresholded_g_pred_probs"] = np.concatenate(
            all_thresholded_g_pred_probs, axis=0
        )

    if return_coords:
        concatenated_results["coords"] = np.concatenate(all_coords, axis=0)

    if return_masks:
        concatenated_results["masks"] = np.concatenate(all_masks, axis=0)

    return concatenated_results
