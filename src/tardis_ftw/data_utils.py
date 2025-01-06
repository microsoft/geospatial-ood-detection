# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing utilities for processing ID and WILD sets."""

import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import kornia.augmentation as K
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
import tqdm
from planetary_computer import sign
from pyproj import Transformer
from pystac_client import Client
from rasterio.mask import mask
from shapely.geometry import box
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import (Path, array_to_tensor,
                                     download_and_extract_archive,
                                     extract_archive)
from torchgeo.transforms import AugmentationSequential


class FieldsOfTheWorldOOD(NonGeoDataset):
    """
    Fields Of The World dataset with support for sampling N images per country.

    Attributes:
        splits (tuple): Dataset splits - "train", "val", and "test".
        targets (tuple): Available target types - "2-class", "3-class", and "instance".
        valid_countries (tuple): List of valid countries in the dataset.
        base_url (str): Base URL for downloading the dataset.
    """

    splits = ("train", "val", "test")
    targets = ("2-class", "3-class", "instance")

    valid_countries = (
        "austria",
        "belgium",
        "brazil",
        "cambodia",
        "corsica",
        "croatia",
        "denmark",
        "estonia",
        "finland",
        "france",
        "germany",
        "india",
        "kenya",
        "latvia",
        "lithuania",
        "luxembourg",
        "netherlands",
        "portugal",
        "rwanda",
        "slovakia",
        "slovenia",
        "south_africa",
        "spain",
        "sweden",
        "vietnam",
    )

    base_url = "https://data.source.coop/kerner-lab/fields-of-the-world-archive/"

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        target: str = "2-class",
        countries: str | Sequence[str] = ["austria"],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
        sample_N: int = 50,
    ) -> None:
        """
        Initialize a Fields Of The World dataset instance.

        Args:
            root (Path): Root directory where the dataset is stored.
            split (str): Dataset split - "train", "val", or "test".
            target (str): Target mask type - "2-class", "3-class", or "instance".
            countries (str | Sequence[str]): Countries to include in the dataset.
            transforms (Callable): Optional transforms to apply to the samples.
            download (bool): If True, downloads the dataset.
            checksum (bool): If True, verifies the MD5 checksum.
            sample_N (int): Number of images to sample per country.
        """
        assert split in self.splits
        assert target in self.targets
        if isinstance(countries, str):
            countries = [countries]
        assert set(countries) <= set(self.valid_countries)

        self.root = root
        self.split = split
        self.target = target
        self.countries = countries
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.sample_N = sample_N

        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index: int) -> dict[str, Tensor | np.ndarray]:
        """
        Return a sample at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, mask, and coordinates.
        """
        win_a_fn = self.files[index]["win_a"]
        win_b_fn = self.files[index]["win_b"]
        mask_fn = self.files[index]["mask"]
        coords = self.files[index]["coords"]

        win_a = self._load_image(win_a_fn)
        win_b = self._load_image(win_b_fn)
        mask = self._load_target(mask_fn)

        image = torch.cat((win_a, win_b), dim=0)
        sample = {"image": image, "mask": mask, "coords": coords}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.files)

    def _load_files(self) -> list[dict[str, str | np.ndarray]]:
        """
        Load file paths and calculate coordinates for the dataset.

        Returns:
            list: List of dictionaries containing file paths and coordinates.
        """
        files = []
        for country in self.countries:
            df = pd.read_parquet(
                os.path.join(self.root, country, f"chips_{country}.parquet")
            )
            aois = df[df["split"] == self.split]["aoi_id"].values
            # Sample N or all available AOIs if fewer than N
            sampled_aois = aois[: min(len(aois), self.sample_N)]

            for aoi in sampled_aois:
                if self.target == "instance":
                    subdir = "instance"
                elif self.target == "2-class":
                    subdir = "semantic_2class"
                elif self.target == "3-class":
                    subdir = "semantic_3class"

                win_a_fn = os.path.join(
                    self.root, country, "s2_images", "window_a", f"{aoi}.tif"
                )
                win_b_fn = os.path.join(
                    self.root, country, "s2_images", "window_b", f"{aoi}.tif"
                )

                # Skip AOIs missing imagery
                if not (os.path.exists(win_a_fn) and os.path.exists(win_b_fn)):
                    continue

                # Calculate coordinates
                coords = self._compute_coords(win_a_fn)

                sample = {
                    "win_a": win_a_fn,
                    "win_b": win_b_fn,
                    "mask": os.path.join(
                        self.root, country, "label_masks", subdir, f"{aoi}.tif"
                    ),
                    "coords": coords,
                }
                files.append(sample)

        return files

    def _compute_coords(self, filepath):
        """
        Compute the center coordinates of an image.

        Args:
            filepath (str): Path to the image file.

        Returns:
            torch.Tensor: Tensor containing latitude and longitude.
        """
        _, bounds, crs = self._load_image(filepath, return_meta=True)
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2

        # Transform coordinates to WGS84 (latitude and longitude)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = transformer.transform(center_x, center_y)
        return torch.tensor([center_lat, center_lon], dtype=torch.float)

    def _load_image(self, path: str, return_meta: bool = False):
        """
        Load an image and convert it to a tensor.

        Args:
            path (str): Path to the image file.
            return_meta (bool): If True, returns image metadata.

        Returns:
            torch.Tensor: Image tensor.
            tuple (optional): Bounds and CRS metadata if `return_meta` is True.
        """
        filename = os.path.join(path)
        with rasterio.open(filename) as dataset:
            array = dataset.read()
            tensor = array_to_tensor(array).float()

            if return_meta:
                bounds = dataset.bounds
                crs = dataset.crs
                return tensor, bounds, crs

        return tensor

    def _load_target(self, path: Path) -> Tensor:
        """
        Load a target mask corresponding to an image.

        Args:
            path (Path): Path to the target mask file.

        Returns:
            torch.Tensor: Target mask tensor.
        """
        filename = os.path.join(path)
        with rasterio.open(filename) as f:
            array = f.read(1)
            tensor = torch.from_numpy(array).long()
        return tensor

    def _verify(self) -> None:
        """
        Verify the integrity of the dataset, downloading it if necessary.
        """
        for country in self.countries:
            if self._verify_data(country):
                continue

            filename = f"{country}.zip"
            pathname = os.path.join(self.root, filename)
            if os.path.exists(pathname):
                extract_archive(pathname, os.path.join(self.root, country))
                continue

            if not self.download:
                raise DatasetNotFoundError(self)

            download_and_extract_archive(
                self.base_url + filename,
                os.path.join(self.root, country),
                filename=filename,
                md5=self.country_to_md5[country] if self.checksum else None,
            )

    def _verify_data(self, country: str) -> bool:
        """
        Verify if the data for a specific country is extracted.

        Args:
            country (str): Name of the country.

        Returns:
            bool: True if the data exists, False otherwise.
        """
        for entry in ["label_masks", "s2_images", f"chips_{country}.parquet"]:
            if not os.path.exists(os.path.join(self.root, country, entry)):
                return False

        return True


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
        """
        Initialize the FTWDataModule.

        Args:
            root_folder_torchgeo (str): Root folder for the dataset.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for data loading.
            train_countries (list[str]): Countries for training.
            val_countries (list[str]): Countries for validation.
            test_countries (list[str]): Countries for testing.
            download (bool): If True, downloads the dataset.
            sample_N_from_each_country (int): Number of samples per country.
            target (str): Target type - "2-class", "3-class", or "instance".
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
        super().__init__(FieldsOfTheWorldOOD, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """
        Set up datasets for training, validation, or testing.

        Args:
            stage (str): Stage of processing - "fit", "validate", or "test".
        """
        if stage in ["fit"]:
            self.train_dataset = FieldsOfTheWorldOOD(
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
            self.val_dataset = FieldsOfTheWorldOOD(
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
            self.test_dataset = FieldsOfTheWorldOOD(
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
        """
        Create DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
        )

    def val_dataloader(self):
        """
        Create DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Create DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for testing.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )


class WILDDataset(Dataset):
    """
    Dataset for WILD files with planting and harvesting images.
    """

    def __init__(self, directory):
        """
        Initialize the WILDDataset dataset.

        Args:
            directory (str): Path to the directory containing WILD images.
        """
        self.directory = directory
        self.paired_files = self._group_files_by_randomid()
        self.valid_pairs, self.coords = self._filter_valid_pairs()

    def _group_files_by_randomid(self):
        """
        Group files by their random ID and ensure they are valid pairs.

        Returns:
            list: List of valid paired file paths.
        """
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
        """
        Compute the center coordinates of an image.

        Args:
            filepath (str): Path to the image file.

        Returns:
            torch.Tensor: Tensor containing latitude and longitude.
        """
        _, bounds, crs = self._load_image(filepath, return_meta=True)
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2

        # Transform coordinates to WGS84 (latitude and longitude)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = transformer.transform(center_x, center_y)
        return torch.tensor([center_lat, center_lon], dtype=torch.float)

    def _filter_valid_pairs(self):
        """
        Filter valid pairs of images where coordinates match.

        Returns:
            tuple: List of valid pairs and their coordinates.
        """
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
        """
        Return the number of valid pairs in the dataset.

        Returns:
            int: Number of valid pairs.
        """
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        """
        Return a paired sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the combined image, coordinates, and file pair.
        """
        file_pair = self.valid_pairs[idx]
        tensors = []
        pair_coords = self.coords[idx]  # Use precomputed coordinates

        for filename in file_pair:
            filepath = os.path.join(self.directory, filename)
            tensor = self._load_image(filepath)
            tensors.append(tensor)

        # Concatenate the tensors along the first dimension (channels)
        combined_tensor = torch.cat(tensors, dim=0)

        sample = {
            "image": combined_tensor,
            "coords": pair_coords,
            "file_pair": file_pair,
        }
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
    return_slc_freq=False,
    upsample=True,
    max_batches=None,
):
    """
    Process batches from the WILD dataset using the provided model.

    Args:
        dataloader: DataLoader for the WILD dataset.
        ood_model: Out-of-Distribution model for predictions.
        return_batch (bool): Return input batch images.
        return_f_pred (bool): Return f-model predictions.
        return_g_pred (bool): Return g-model predictions.
        return_thresholded_g_pred (bool): Return thresholded g-model predictions.
        return_coords (bool): Return coordinates.
        return_slc_freq (bool): Return SCL frequency statistics.
        upsample (bool): Whether to upsample predictions.
        max_batches (int): Maximum number of batches to process.

    Returns:
        dict: Dictionary containing processed results.
    """

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

        if return_slc_freq:
            filename = batch["file_pair"][0]
            slc_freq = query_and_get_scl_multiple(
                dataloader.dataset.directory, filename
            )
            results["slc_freq"] = slc_freq

        # Clear GPU memory
        del images, f_preds, g_pred_probs
        torch.cuda.empty_cache()

        return results

    all_f_preds = []
    all_g_pred_probs = []
    all_thresholded_g_pred_probs = []
    all_coords = []
    all_batches = []
    all_slc_freq = []

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

        if return_slc_freq:
            all_slc_freq.append(results["slc_freq"])

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

    if return_slc_freq:
        concatenated_results["slc_freq"] = np.concatenate(all_slc_freq, axis=0)

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
    """
    Process batches from the ID dataset using the provided model.

    Args:
        dataloader: DataLoader for the ID dataset.
        ood_model: Out-of-Distribution model for predictions.
        return_batch (bool): Return input batch images.
        return_f_pred (bool): Return f-model predictions.
        return_g_pred (bool): Return g-model predictions.
        return_thresholded_g_pred (bool): Return thresholded g-model predictions.
        return_coords (bool): Return coordinates.
        return_masks (bool): Return ground-truth masks.
        upsample (bool): Whether to upsample predictions.
        max_batches (int): Maximum number of batches to process.

    Returns:
        dict: Dictionary containing processed results.
    """
    np.set_printoptions(suppress=True)

    def process_batch(batch):
        images = batch["image"].to("cuda")
        masks = batch["mask"].to("cuda")
        coords = batch["coords"]

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
            results["coords"] = coords.numpy()

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


def query_and_get_scl_multiple(
    directory, filenames, output_dir="downloads", return_slc_band=False
):
    """
    Query Planetary Computer for SCL layers and download/clips them to match patches.

    Args:
        directory (str): Directory containing Sentinel-2 patch files.
        filenames (list[str]): List of patch filenames.
        output_dir (str): Directory to save clipped SCL files.
        return_slc_band (bool): If True, returns SCL bands.

    Returns:
        list: Statistics of SCL class frequencies for all patches.
    """
    # Prepare results
    clipped_scl = []
    all_stats = []

    # Planetary Computer API URL
    api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = Client.open(api_url)

    for filename in filenames:
        try:
            # Construct full file path
            patch_file = os.path.join(directory, filename)

            # Step 1: Extract metadata from patch
            with rasterio.open(patch_file) as patch:
                patch_bounds = patch.bounds
                patch_crs = patch.crs
                patch_geometry = box(*patch_bounds)

            # Convert bounds to WGS84 (Lat/Lon) for the query
            transformer = Transformer.from_crs(patch_crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(patch_bounds[0], patch_bounds[1])
            max_lon, max_lat = transformer.transform(patch_bounds[2], patch_bounds[3])
            bbox = [min_lon, min_lat, max_lon, max_lat]

            # Search for Sentinel-2 L2A items
            search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime="2024-01-01/2024-12-31",
            )

            # Get the first matching STAC Item
            items = list(search.items())
            if not items:
                raise ValueError(
                    f"No matching Sentinel-2 scenes found for patch: {filename}"
                )
            item = items[0]
            signed_item = sign(item)

            # Get the SCL asset URL
            scl_href = signed_item.assets["SCL"].href

            # Download the SCL layer
            response = requests.get(scl_href, stream=True)
            os.makedirs(output_dir, exist_ok=True)
            scl_file = os.path.join(output_dir, f"{filename}_scl_layer.tif")
            with open(scl_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            # Clip the SCL layer to the patch extent
            with rasterio.open(scl_file) as scl:
                # Reproject geometry if necessary
                if patch_crs != scl.crs:
                    transformer = Transformer.from_crs(
                        patch_crs, scl.crs, always_xy=True
                    )
                    patch_geometry = box(
                        *transformer.transform(*patch_bounds[0:2]),
                        *transformer.transform(*patch_bounds[2:4]),
                    )

                # Mask SCL data using the patch geometry
                clipped_scl, _ = mask(
                    scl, [patch_geometry], crop=True, all_touched=True
                )

            # Calculate SCL class statistics
            scl_values = clipped_scl[0].flatten()
            scl_values = scl_values[scl_values > 0]  # Exclude NoData values
            unique_classes, counts = np.unique(scl_values, return_counts=True)
            stats = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}

            print(stats)
            all_stats.append(stats)

        except Exception as e:
            print(f"Error processing patch {filename}: {e}")
            all_stats.append(None)  # Append None for the failed patch

    if return_slc_band:
        return clipped_scl, all_stats
    return all_stats
