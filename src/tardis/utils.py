# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities."""

import json
import os
import random
import warnings

import folium
import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from kornia.augmentation import AugmentationSequential
from pyproj import Proj, transform
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torchgeo.datamodules.eurosat import SPATIAL_MEAN, SPATIAL_STD

warnings.simplefilter(action="ignore", category=FutureWarning)


def set_seed(seed):
    """
    Set the random seed for reproducibility across libraries.

    Args:
        seed (int): The random seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_exp_ev_metrics_json(
    exp_name,
    base_dir,
):
    """
    Load experiment evaluation metrics from a JSON file.

    Args:
        exp_name (str): Name of the experiment.
        base_dir (str): Base directory containing experiment results (default: "geospatial-ood-detection/experiments").

    Returns:
        dict: A dictionary containing experiment results.
    """
    results_path = os.path.join(
        base_dir, exp_name, "test_eval_results_{}.json".format(exp_name)
    )
    print("Reading experiment from: ", results_path)

    with open(results_path, "r") as f:
        results = json.load(f)
    return results

def get_all_layer_names(model):
    """
    Retrieve the names of all layers in a model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        list: A list of layer names.
    """
    layer_names = []
    for name, _ in model.named_modules():
        layer_names.append(name)
    return layer_names


def get_layer_shapes(model, input_tensor):
    """
    Retrieve the input and output shapes of all layers in a model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): Example input tensor for the model.

    Returns:
        list: A list of tuples containing layer names, input shapes, and output shapes.
    """
    layer_shapes = []

    def hook_fn(module, input, output):
        layer_full_name = layer_name_map[module]
        input_shape = tuple(input[0].shape)
        output_shape = tuple(np.array(output).shape)
        layer_shapes.append((layer_full_name, input_shape, output_shape))

    hooks = []
    layer_name_map = {}
    for name, layer in model.named_modules():
        if (
            not isinstance(layer, nn.Sequential)
            and not isinstance(layer, nn.ModuleList)
            and not (layer == model)
        ):
            hooks.append(layer.register_forward_hook(hook_fn))
            layer_name_map[layer] = name

    model(input_tensor)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return layer_shapes


def get_conv_layer_names(model, layer_shapes):
    """
    Extract the names of all convolutional layers in a model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_shapes (list): List of tuples containing layer shapes.

    Returns:
        list: A list of convolutional layer names.
    """
    conv_layers = []
    for layer_full_name, _, _ in layer_shapes:
        if isinstance(getattr(model, layer_full_name), nn.Conv2d):
            conv_layers.append(layer_full_name)
    return conv_layers


def extract_rgb(image, channel_order=(3, 2, 1)):
    """
    Extract and stack RGB channels from a Sentinel-2 image tensor.

    Args:
        image (torch.Tensor): Multi-channel image tensor.
        channel_order (tuple): Channel indices for R, G, and B (default: (3, 2, 1)).

    Returns:
        torch.Tensor: RGB image tensor in (H, W, C) format.
    """
    rgb_image = image[channel_order, :, :]
    return rgb_image.permute(1, 2, 0)


def extract_rgb_np(image, channel_order=(3, 2, 1)):
    """
    Extract and stack RGB channels from a multi-channel numpy image.

    Args:
        image (np.ndarray): Multi-channel image array.
        channel_order (tuple): Channel indices for R, G, and B (default: (3, 2, 1)).

    Returns:
        np.ndarray: RGB image array in (H, W, C) format.
    """
    rgb_image = image[channel_order, :, :]
    return np.transpose(
        rgb_image, (1, 2, 0)
    )


def percentile_stretch(image, p1=1, p99=99):
    """
    Apply percentile stretching to an image for normalization.

    Args:
        image (np.ndarray): Input image array.
        p1 (int): Lower percentile for stretching (default: 1).
        p99 (int): Upper percentile for stretching (default: 99).

    Returns:
        np.ndarray: Stretched image normalized to the range [0, 1].
    """
    # Compute the lower and upper percentile values
    vmin = np.percentile(image, p1)
    vmax = np.percentile(image, p99)

    # Clip the image to the percentile range and scale to [0, 1]
    image_stretched = np.clip(image, vmin, vmax)
    image_stretched = (image_stretched - vmin) / (vmax - vmin)

    return image_stretched


def collect_data_to_save(dataloader, device, return_filename=True):
    """
    Collect images, labels, and filenames from a DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for batch processing.
        device (str): Device to move the tensors to.
        return_filename (bool): If True, returns filenames (default: True).

    Returns:
        tuple: Arrays containing images, labels, and optionally filenames.
    """
    images_list = []
    labels_list = []
    filenames_list = []

    for _, sample in tqdm.tqdm(enumerate(dataloader)):
        images = sample["image"].to(device)
        labels = sample["label"].to(device)

        images_list.append(images.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())

        if return_filename:
            files = sample["filename"]
            filenames_list.extend(files)
            filenames_array = np.array(filenames_list)

    images_array = np.concatenate(images_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    if return_filename:
        return images_array, labels_array, filenames_array
    else:
        return images_array, labels_array


def load_data(data_file, return_filename=True):
    """
    Load image and label data from a `.npz` file.

    Args:
        data_file (str): Path to the `.npz` file.
        return_filename (bool): If True, returns filenames (default: True).

    Returns:
        tuple: Arrays containing images, labels, and optionally filenames.
    """
    data = np.load(data_file, allow_pickle=True)
    images = data["images"]
    labels = data["labels"]
    if return_filename:
        filenames = data["filenames"]
        return images, labels, filenames
    else:
        return images, labels


def compute_overall_statistics(images):
    """
    Compute the overall mean and standard deviation for an image dataset.

    Args:
        images (array-like): Input image dataset.

    Returns:
        tuple: Mean and standard deviation of the flattened image dataset.
    """
    flat_images = images.flatten()
    mean = np.mean(flat_images)
    std = np.std(flat_images)
    return mean, std


def plot_data_distributions(train_data, val_data, test_data, title_prefix):
    """
    Plot data distributions for train, validation, and test datasets.

    Args:
        train_data (array-like): Training dataset.
        val_data (array-like): Validation dataset.
        test_data (array-like): Test dataset.
        title_prefix (str): Prefix for the plot titles.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    sns.kdeplot(train_data, color="blue", label="Train Data", fill=True)
    sns.kdeplot(val_data, color="green", label="Validation Data", fill=True)
    sns.kdeplot(test_data, color="red", label="Test Data", fill=True)
    plt.title(f"{title_prefix} -- Mean and Std Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    df = pd.DataFrame({"Train": train_data, "Validation": val_data, "Test": test_data})

    plt.figure(figsize=(12, 6))
    sns.histplot(df, kde=True, stat="density", element="step", fill=True)
    plt.title(f"{title_prefix} -- Overall Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


def collect_layer_activations(
    model,
    dm,
    dataloader,
    layer_names,
    device,
    n_batches_to_process,
    getitem_keys,
    collect_aug_input,
    collect_annot_preds=False,
):
    """
    Collect activations from specified layers of a model during forward passes.

    This function uses the `LayerActivationExtractor` class to register hooks on specified layers 
    and extract activations as the model processes batches from a DataLoader.

    Args:
        model (torch.nn.Module): The PyTorch model to extract activations from.
        dm: Data module that provides augmentations for the input data.
        dataloader (torch.utils.data.DataLoader): DataLoader to iterate over batches of input data.
        layer_names (list): List of layer names from which to collect activations.
        device (str): Device to run the model on ("cuda" or "cpu").
        n_batches_to_process (int): Number of batches to process from the DataLoader.
        getitem_keys (list): Keys to extract the relevant data (e.g., image, label) from the batch.
        collect_aug_input (bool): If True, collects augmented inputs in addition to activations.
        collect_annot_preds (bool): If True, collects annotations (ground truth) and model predictions.

    Returns:
        dict: A dictionary containing the following keys:
            - "activations" (dict): Activations from the specified layers.
            - "annotations" (np.ndarray, optional): Ground truth labels (if `collect_annot_preds` is True).
            - "predictions" (np.ndarray, optional): Model predictions (if `collect_annot_preds` is True).
            - "aug_inputs" (np.ndarray, optional): Augmented input tensors (if `collect_aug_input` is True).
    """

    extractor = LayerActivationExtractor(model, device)
    return extractor.collect_activations(
        dm,
        dataloader,
        layer_names,
        getitem_keys,
        n_batches_to_process,
        collect_annot_preds,
        collect_aug_input,
    )


class LayerActivationExtractor:
    def __init__(self, model, device):
        """
        Extract activations from specific layers of a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.model = model.to(device)
        self.device = device
        self.activations = {}
        self.hook_handles = []

    def _register_hooks(self, layer_names):
        def get_activation(name):
            def hook(module, input, output):
                # Store activations for each sample individually
                batch_activations = output.detach().cpu().numpy()
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].extend(
                    batch_activations
                )  # Store each sample's activations

            return hook

        for name, layer in self.model.named_modules():
            if name in layer_names:
                handle = layer.register_forward_hook(get_activation(name))
                self.hook_handles.append(handle)

    def _clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def collect_activations(
        self,
        dm,
        dataloader,
        layer_names,
        getitem_keys,
        n_batches_to_process=5,
        collect_annot_preds=False,
        collect_aug_input=False,
        device="cuda",
    ):
        """
        Collect activations from specified layers during forward passes.

        Args:
            dm: Data module for augmentations.
            dataloader (torch.utils.data.DataLoader): DataLoader to process batches.
            layer_names (list): List of layer names to collect activations from.
            getitem_keys (list): Keys to extract data from batches.
            n_batches_to_process (int): Number of batches to process.
            collect_annot_preds (bool): If True, collects predictions and annotations.
            collect_aug_input (bool): If True, collects augmented input tensors.
            device (str): Device to perform computations on.

        Returns:
            dict: Dictionary containing activations, annotations, predictions, and optionally augmented inputs.
        """

        self.model.eval()
        self.activations = {}
        self.hook_handles = []
        self._register_hooks(layer_names)

        if collect_annot_preds:
            annotations = []
            predictions = []

        if collect_aug_input:
            aug_inputs = []

        with torch.no_grad():
            for batch_idx, batch in tqdm.tqdm(
                enumerate(dataloader), total=n_batches_to_process
            ):
                if (
                    n_batches_to_process is not None
                    and batch_idx >= n_batches_to_process
                ):
                    break

                aug_batch = dm.test_aug(batch)
                labels = batch[getitem_keys[1]]
                aug_input = aug_batch[getitem_keys[0]].to(device)
                logits = self.model(aug_input)

                if collect_annot_preds:
                    labels = sample[getitem_keys[1]].to(self.device)
                    annotations.extend(labels.cpu().numpy())
                    predictions.extend(logits.cpu().numpy())

                if collect_aug_input:
                    aug_inputs.append(aug_input.cpu())

        self._clear_hooks()

        numpy_activations = convert_activations_to_numpy(self.activations)

        results = {"activations": numpy_activations}

        if collect_annot_preds:
            results["annotations"] = np.array(annotations)
            results["predictions"] = np.array(predictions)

        if collect_aug_input:
            concatenated_aug_inputs = torch.cat(aug_inputs, dim=0)
            results["aug_inputs"] = concatenated_aug_inputs.numpy()

        return results


def convert_activations_to_numpy(activations):
    """
    Convert activations collected during forward passes into numpy arrays.

    Args:
        activations (dict): Dictionary containing layer activations as lists of tensors.

    Returns:
        dict: Dictionary with layer names as keys and numpy arrays of activations as values.
    """
    numpy_activations = {}
    for name, acts in activations.items():
        numpy_activations[name] = np.array(
            acts
        )
    return numpy_activations


def downsample_activations(activations, method="mean_std"):
    """
    Downsample activations using specified methods.

    Args:
        activations (dict): Dictionary containing activations for each layer.
        method (str): Downsampling method - 'mean_std', 'avg_pool', 'max_pool', or 'nodownsample'.

    Returns:
        dict: Dictionary with downsampled activations.
    """
    if method == "nodownsample":
        pca = PCA(n_components=10)

    downsampled_activations = {}
    for layer_name, layer_activations_tensor in activations.items():
        print(
            "Before downsampling, activations shape for layer",
            layer_name,
            ":",
            layer_activations_tensor.shape,
        )
        layer_activations_tensor = torch.tensor(
            layer_activations_tensor, dtype=torch.float
        )

        if method == "mean_std":
            if layer_activations_tensor.ndim == 4:
                mean = layer_activations_tensor.mean(axis=(2, 3))
                std = layer_activations_tensor.std(axis=(2, 3))
                mean, _ = mean.max(axis=-1)
                std, _ = std.max(axis=-1)
                downsampled_activations[layer_name] = (mean, std)
            elif layer_activations_tensor.ndim == 2:
                mean = layer_activations_tensor.mean(axis=1)
                std = layer_activations_tensor.std(axis=1)
                downsampled_activations[layer_name] = (mean, std)
            else:
                raise ValueError(
                    "Unsupported number of dimensions. Supported dimensions are 2 (B, C) and 4 (B, C, H, W)."
                )

        elif method == "avg_pool":
            # Apply global average pooling
            avg_pooled_activations = F.adaptive_avg_pool2d(
                layer_activations_tensor, (1, 1)
            )
            avg_pooled_activations = avg_pooled_activations.view(
                avg_pooled_activations.size(0), -1
            )
            downsampled_activations[layer_name] = avg_pooled_activations.numpy()
        elif method == "max_pool":
            # Apply global max pooling
            max_pooled_activations = F.adaptive_max_pool2d(
                layer_activations_tensor, (1, 1)
            )
            max_pooled_activations = max_pooled_activations.view(
                max_pooled_activations.size(0), -1
            )
            downsampled_activations[layer_name] = max_pooled_activations.numpy()
        elif method == "nodownsample":
            # Apply PCA
            B, C, H, W = layer_activations_tensor.shape
            layer_act_b_x = layer_activations_tensor.reshape(B, C * H * W)
            reduced_data = pca.fit_transform(layer_act_b_x)
            downsampled_activations[layer_name] = reduced_data
        else:
            raise ValueError(f"Unsupported downsampling method: {method}")
    return downsampled_activations


def collect_and_process_activations(
    model,
    dm,
    dataloader,
    layer_names,
    device,
    n_batches_to_process,
    getitem_keys,
    collect_aug_input,
    downsample_method="mean_std",
    verbose=False,
):
    """
    Collect and process activations from specific model layers using a DataLoader.

    This function collects layer activations, downsamples them using a specified method, and structures them into 
    a feature matrix.

    Args:
        model (torch.nn.Module): The PyTorch model to extract activations from.
        dm: Data module providing augmentations.
        dataloader (torch.utils.data.DataLoader): DataLoader to iterate over input batches.
        layer_names (list): List of layer names to collect activations from.
        device (str): Device to run the model on ("cuda" or "cpu").
        n_batches_to_process (int): Number of batches to process from the DataLoader.
        getitem_keys (list): Keys to extract relevant inputs from the data batches.
        collect_aug_input (bool): If True, collects augmented inputs.
        downsample_method (str): Method to downsample activations ("mean_std", "avg_pool", "max_pool", etc.).
        verbose (bool): If True, prints progress and debug information.

    Returns:
        tuple:
            - dict: Structured activations with processed outputs for each layer.
            - dict: Property lengths corresponding to the downsampled activations.
            - np.ndarray (optional): Augmented inputs, if `collect_aug_input` is True.
    """

    # Step 1: Collect activations
    activations_results = collect_layer_activations(
        model,
        dm,
        dataloader,
        layer_names,
        device,
        n_batches_to_process,
        getitem_keys,
        collect_aug_input,
        collect_annot_preds=False,
    )

    # Step 2: Downsample the collected activations
    downsampled_activations = downsample_activations(
        activations_results["activations"], method=downsample_method
    )

    # Step 3: Structure the activations
    structured_activations = {}
    property_lengths = {}  # Initialize the property lengths dictionary

    for layer_name in layer_names:
        if downsample_method == "mean_std":
            means, stds = downsampled_activations[layer_name]

            mean_array = np.array(means)
            std_array = np.array(stds)

            mean_array = np.expand_dims(mean_array, axis=-1)
            std_array = np.expand_dims(std_array, axis=-1)

            concatenated = np.concatenate([mean_array, std_array], axis=-1)
            structured_activations[layer_name] = {"activations": concatenated}

            property_lengths[layer_name] = {"activations": concatenated.shape[-1]}
        else:
            # For pooling methods
            activations = downsampled_activations[layer_name]
            structured_activations[layer_name] = {"activations": activations}

            # The length is determined by the second dimension of the activations after downsampling -- (B, C)
            property_lengths[layer_name] = {"activations": activations.shape[1]}

    if verbose:
        for layer_name in layer_names:
            print(
                f"{layer_name} ({downsample_method}):",
                structured_activations[layer_name]["activations"].shape,
            )
        print("Property lengths:", property_lengths)

    if collect_aug_input:
        aug_inputs = activations_results["aug_inputs"]
        return structured_activations, property_lengths, aug_inputs

    return structured_activations, property_lengths


def create_feature_matrix_and_labels(
    model,
    dm,
    train_dataloader,
    test_dataloader,
    layer_names,
    device,
    getitem_keys,
    n_batches_to_process=5,
    downsample_method="mean_std",
    collect_aug_input=False,
    verbose=False,
):
    """
    Create a feature matrix and corresponding labels from the activations of a given model.

    Args:
        model (torch.nn.Module): The model to collect activations from.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        layer_names (list): A list of layer names to collect activations from.
        device (torch.device): The device to use for computation.
        n_batches_to_process (int, optional): The number of batches to process from each dataloader. Defaults to 0, which processes all batches.
        downsample_method (str, optional): The method to downsample the activations. Defaults to 'mean_std'.
        verbose (bool, optional): Whether to print progress information. Defaults to False.

    Returns:
        tuple: A tuple containing the feature matrix (X) and the corresponding labels (y).
    """

    X, y = [], []

    # Process train dataloader
    train_activations, train_property_lengths, train_images = (
        collect_and_process_activations(
            model=model,
            dm=dm,
            dataloader=train_dataloader,
            layer_names=layer_names,
            device=device,
            n_batches_to_process=n_batches_to_process,
            getitem_keys=getitem_keys,
            downsample_method=downsample_method,
            verbose=verbose,
            collect_aug_input=collect_aug_input,
        )
    )

    for layer_name in layer_names:
        for sample_activations in train_activations[layer_name]["activations"]:
            X.append(sample_activations.flatten())  # Flatten and append
        y.extend([0] * len(train_activations[layer_name]["activations"]))  # Train label

    # Process test dataloader
    test_activations, test_property_lengths, test_images = (
        collect_and_process_activations(
            model=model,
            dm=dm,
            dataloader=test_dataloader,
            layer_names=layer_names,
            device=device,
            n_batches_to_process=n_batches_to_process,
            getitem_keys=getitem_keys,
            downsample_method=downsample_method,
            verbose=verbose,
            collect_aug_input=collect_aug_input,
        )
    )

    for layer_name in layer_names:
        for sample_activations in test_activations[layer_name]["activations"]:
            X.append(sample_activations.flatten())
        y.extend([1] * len(test_activations[layer_name]["activations"]))

    assert len(X) == len(y), "X and y should have the same length."
    assert (
        train_property_lengths == test_property_lengths
    ), "Train and test property lengths should match."

    return np.array(X), np.array(y), train_property_lengths, train_images, test_images


inference_aug = AugmentationSequential(
    K.Normalize(
        mean=list(SPATIAL_MEAN.values()),
        std=list(SPATIAL_STD.values()),
        p=1.0,
        keepdim=True,
    ),
    data_keys=["image"],
    same_on_batch=True,
)

class GeoTIFFDataset(Dataset):
    def __init__(self, root_dir, split_dir, split="test", transform=None):
        """
        Custom PyTorch Dataset for loading GeoTIFF images.

        Args:
            root_dir (str): Root directory containing the dataset.
            split_dir (str): Directory containing the split files.
            split (str): Dataset split - "train", "val", or "test".
            transform (callable): Transformations to apply to the images.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.samples = []

        self._load_split_data(split_dir)

    def _load_split_data(self, split_dir):
        """
        Load file paths for the specified data split (train, validation, or test).

        Args:
            split_dir (str): Directory containing split files for the dataset.

        Raises:
            FileNotFoundError: If the specified split file does not exist.

        Returns:
            None
        """
        split_file = f"{split_dir}/eurosat-{self.split}.txt"
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"No such file for split data: {split_file}")

        with open(split_file, "r") as file:
            for line in file:
                filename = line.strip()
                class_label = filename.split("_")[
                    0
                ]
                full_path = os.path.join(self.root_dir, class_label, filename)
                if os.path.exists(full_path.replace(".jpg", ".tif")):
                    self.samples.append(
                        (full_path.replace(".jpg", ".tif"), class_label)
                    )

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing the image, label, and file path.
        """

        path_class = self.samples[idx]
        image = self._load_image(path_class[0])

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": path_class[1], "path": path_class[0]}

    def _load_image(self, path):
        """
        Load a GeoTIFF image as a tensor.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: Image tensor.
        """

        with rasterio.open(path) as src:
            array = src.read().astype("float32")
        tensor = torch.from_numpy(array)
        return tensor


def process_and_classify_samples(
    layer_name,
    split,
    dataloader,
    model,
    classifier,
    device,
    max_batches,
):
    """
    Extract activations, classify samples, and collect results.

    Args:
        layer_name (str): Name of the layer to extract activations from.
        split (str): Dataset split - "train", "val", or "test".
        dataloader (torch.utils.data.DataLoader): DataLoader for input samples.
        model (torch.nn.Module): PyTorch model for inference.
        classifier (object): Classifier for predictions.
        device (str): Device for computations.
        max_batches (int): Maximum number of batches to process.

    Returns:
        pd.DataFrame: DataFrame containing results with predictions, labels, and geolocation data.
    """

    numeric_gt_labels_list = []
    clsf_pred_list = []
    ood_predictions_list = []
    lat_lon_list = []
    filename_list = []

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            if max_batches is not None and i >= max_batches:
                break

            inputs = batch["image"].to(device)
            gt_labels = batch["label"]
            gt_labels = [int(label) for label in gt_labels]
            paths = batch["path"]

            activations, predicted_cls = extract_activations(
                model, inputs, layer_name, return_predicted_class=True
            )

            predicted_cls = predicted_cls.detach().cpu().numpy()

            dsampled = downsample_activations_for_ood_on_map(activations, layer_name)[
                layer_name
            ]

            ood_predictions = classifier.predict(dsampled)

            # Extract geographic information
            geo_data = [get_coordinates_from_tiff(path) for path in paths]
            lat_lon = [f"{lat}, {lon}" for lat, lon, _ in geo_data]
            filenames = [filename for _, _, filename in geo_data]

            numeric_gt_labels_list.extend(gt_labels)
            clsf_pred_list.extend(predicted_cls)
            ood_predictions_list.extend(ood_predictions)
            lat_lon_list.extend(lat_lon)
            filename_list.extend(filenames)

    results = pd.DataFrame(
        {
            "numeric_gt_labels": numeric_gt_labels_list,
            "clsf_pred": clsf_pred_list,
            "ood_predictions": ood_predictions_list,
            "split": split,
            "lat_lon": lat_lon_list,
            "filename": filename_list,
        }
    )

    return results


def get_coordinates_from_tiff(tiff_path):
    """
    Extract geographic coordinates from a GeoTIFF file.

    Args:
        tiff_path (str): Path to the TIFF file.

    Returns:
        tuple: Latitude, longitude, and filename of the TIFF file.
    """

    with rasterio.open(tiff_path) as src:
        meta = src.meta
        proj_lat = meta["transform"][5]
        proj_lon = meta["transform"][2]
        crs_proj = src.crs

    projection = Proj(crs_proj)
    wgs84 = Proj(proj="latlong", datum="WGS84")

    # Convert the coordinates to latitude and longitude in degrees
    lon, lat = transform(projection, wgs84, proj_lon, proj_lat)

    filename = os.path.basename(tiff_path).split(".")[0]
    return lat, lon, filename


def get_activation_hook(activations, layer_name):
    def hook(model, input, output):
        activations[layer_name] = output.detach()
    return hook


def extract_activations(model, inputs, layer_name, return_predicted_class=True):
    """
    Extract activations from a specific layer of a model during forward passes.

    Args:
        model (torch.nn.Module): The PyTorch model.
        inputs (torch.Tensor): Input tensor for the model.
        layer_name (str): Name of the layer to extract activations from.
        return_predicted_class (bool): If True, returns predicted class logits as well.

    Returns:
        tuple: Contains activations from the specified layer and predicted class logits (if `return_predicted_class` is True).
    """
    activations = {}

    # Register hook to the specified layer
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        handle = layer.register_forward_hook(
            get_activation_hook(activations, layer_name)
        )
    else:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # Run the model to get activations
    model.eval()
    with torch.no_grad():
        logits = model(inputs)

    # Remove the hook after use
    handle.remove()

    if return_predicted_class:
        return activations[layer_name], torch.argmax(logits, dim=1)
    # Return the activations from the specified layer
    return activations[layer_name]


def save_results_to_csv(results, base_name, filename):
    """
    Save results to a CSV file.

    Args:
        results (pd.DataFrame): DataFrame containing the results to save.
        base_name (str): Directory path where the file will be saved.
        filename (str): Name of the CSV file.

    Returns:
        None
    """
    csv_path = os.path.join(base_name, filename)
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


def downsample_activations_for_ood_on_map(layer_activations, layer_name):
    """
    Downsample activations of a given layer using average pooling.

    Args:
        layer_activations (torch.Tensor): Activations for a specific layer.
        layer_name (str): Name of the layer.

    Returns:
        dict: A dictionary with the downsampled activations.
    """
    downsampled_activations = {}

    avg_pooled_activations = F.adaptive_avg_pool2d(layer_activations, (1, 1))
    avg_pooled_activations = (
        avg_pooled_activations.view(avg_pooled_activations.size(0), -1).cpu().numpy()
    )
    downsampled_activations[layer_name] = avg_pooled_activations
    return downsampled_activations


def mean_std_concat(raw_downsampled_activations, layer):
    """
    Concatenate the mean and standard deviation of activations along the last axis.

    Args:
        raw_downsampled_activations (dict): Dictionary containing mean and standard deviation activations.
        layer (str): Layer name for which activations are processed.

    Returns:
        np.ndarray: Concatenated array with shape (B, 2), where B is the batch size.
    """
    means, stds = raw_downsampled_activations[layer]
    mean_array = np.array(means)
    std_array = np.array(stds)
    # Reshape to (B, 1) if they are (B,)
    mean_array = (
        np.expand_dims(mean_array, axis=-1) if mean_array.ndim == 1 else mean_array
    )
    std_array = np.expand_dims(std_array, axis=-1) if std_array.ndim == 1 else std_array
    # Concatenate along the last axis to get shape (B, 2)
    concatenated = np.concatenate([mean_array, std_array], axis=-1)
    return concatenated


def save_html_map(samples, base_directory, map_name):
    """
    Save an HTML map with markers for geospatial data samples.

    Args:
        samples (pd.DataFrame): DataFrame containing coordinates, predictions, and split labels.
        base_directory (str): Directory to save the HTML map.
        map_name (str): Name of the HTML file to save.

    Returns:
        None
    """
    # Initialize the map at a central location with appropriate zoom level
    m = folium.Map(location=[40, 20], zoom_start=3, tiles="OpenStreetMap")

    # Define colors for binary results and splits
    binary_colors = {0: "red", 1: "green"}
    split_colors = {"train": "purple", "val": "blue", "test": "black"}

    # Process each sample
    for _, row in samples.iterrows():
        lat, lon = map(float, row["lat_lon"].split(","))
        binary_color = binary_colors[row["ood_predictions"]]
        split_color = split_colors[row["split"]]

        # Circle marker for binary result
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=binary_color,
            fill=True,
            fill_color=binary_color,
            fill_opacity=0.7,
            popup=f"File: {row['filename']}<br>Result: {row['ood_predictions']}<br>Split: {row['split']}",
        ).add_to(m)

        # Square marker for the split
        folium.RegularPolygonMarker(
            location=[lat, lon],
            number_of_sides=4,
            radius=10,
            color=split_color,
            fill_opacity=0,
            weight=2,
        ).add_to(m)

    legend_html = """
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 150px; height: 120px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white;
                 ">
     &nbsp; <b>Legend</b> <br>
     &nbsp; <i class="fa fa-circle" style="color:red"></i> OOD Prediction 0 <br>
     &nbsp; <i class="fa fa-circle" style="color:green"></i> OOD Prediction 1 <br>
     &nbsp; <i class="fa fa-square" style="color:purple"></i> Train <br>
     &nbsp; <i class="fa fa-square" style="color:blue"></i> Val <br>
     &nbsp; <i class="fa fa-square" style="color:black"></i> Test <br>
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))

    html_filename = os.path.join(base_directory, f"{map_name}.html")
    m.save(html_filename)


def evaluate_model(model, dm, dataloader, device):
    """
    Evaluate a model using a DataLoader and return predictions and ground truth labels.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dm: Data module that provides augmentations for input data.
        dataloader (torch.utils.data.DataLoader): DataLoader to iterate over evaluation batches.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        tuple: 
            - list: Ground truth labels for the evaluation data.
            - list: Model predictions for the evaluation data.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            aug_batch = dm.test_aug(batch)
            labels = batch["label"]
            aug_input = aug_batch["image"].to(device)
            logits = model(aug_input)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def map_category_to_index(category):
    """
    Map a category name to its index in a predefined list.

    Args:
        category (str): The category name to map.

    Returns:
        int: Index of the category in the predefined list.

    Raises:
        ValueError: If the category is not found in the list.
    """

    categories = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]

    if category not in categories:
        print(f"Category not found in the list of categories: {category}")
        raise ValueError("Category not found in the list of categories.")

    return categories.index(category)
