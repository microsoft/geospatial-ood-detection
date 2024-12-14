# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing the TARDIS wrapper."""

import joblib
import kornia.augmentation as K
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from kornia.constants import Resample
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_model_layers(model: nn.Module) -> list:
    """
    Utility method to return the list of layers in a model.
    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        list: A list of layer names in the model.
    """
    return [name for name, _ in model.named_modules()]


def hook_layer(model: nn.Module, layer_name: str, hook_fn):
    """
    Hook a function to a specific layer of the model.
    Args:
        model (nn.Module): The PyTorch model.
        layer_name (str): The name of the layer to hook.
        hook_fn (callable): The function to execute when the layer is called.

    Returns:
        hook: The registered hook.
    """
    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)
    return hook


class OODModelWrapper(nn.Module):
    """
    A wrapper class that enhances a base model with out-of-distribution (OOD) detection capabilities.

    Args:
        base_model (nn.Module): The original PyTorch model.
        hook_layer_name (str): The name of the layer to hook for feature extraction.
        id_tensor (torch.Tensor): The tensor representing the ID samples.
        wild_tensor (torch.Tensor): The tensor representing the unknown/OOD samples.
        num_clusters (int): The number of clusters to use in KMeans.
        random_state (int): The random state for reproducibility.
        n_estimators (int): The number of estimators for the RandomForestClassifier.
        test_size (float): The proportion of the dataset to include in the test split.
        M (float): The threshold for determining OOD clusters.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hook_layer_name: str,
        main_loader,
        id_loader,
        wild_loader,
        n_batches_to_process: int,
        test_size: float,
        use_optuna: bool,
        num_clusters: int,
        M: float,
        random_state: int,
        n_estimators: int,
        resize_factor: int,
        patch_size: int,
        device="cuda",
        classifier_save_path: str = "ood_classifier.pkl",
    ):
        super(OODModelWrapper, self).__init__()
        self.base_model = base_model
        self.hooked_features = []
        self.hook_handle = None
        self.hook_layer_name = hook_layer_name
        self.main_loader = main_loader
        self.id_loader = id_loader
        self.wild_loader = wild_loader
        self.n_batches_to_process = n_batches_to_process
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.M = M
        self.use_optuna = use_optuna
        self.ood_classifier = None
        self.X = None
        self.y_clustered = None
        self.ood_classifier_metrics = None
        self.classifier_save_path = classifier_save_path
        self.device = device

        self.id_coords = []
        self.wild_coords = []

        self.up_sample = K.Resize(
            (patch_size * resize_factor, patch_size * resize_factor)
        ).to(device)
        self.down_sample = K.Resize(
            (patch_size, patch_size), resample=Resample.NEAREST.name
        ).to(device)

        self.configure_hook()
        self.load_classifier()

    def configure_hook(self):
        """Configure a hook on the specified layer to extract features during the forward pass."""

        def hook_fn(module, input, output):
            self.hooked_features.append(output)

        if self.hook_layer_name:
            self.hook_handle = hook_layer(
                self.base_model, self.hook_layer_name, hook_fn
            )

    def forward(self, x):
        """Forward pass through the base model."""
        return self.base_model(x)

    def get_id_coords(self):
        return self.id_coords

    def get_wild_coords(self):
        return self.wild_coords

    def compute_features(self):
        """
        Compute and return the features extracted from the hooked layer for both ID and WILD samples.

        Args:
            id_loader (DataLoader): DataLoader for ID (in-distribution) samples.
            wild_loader (DataLoader): DataLoader for Unknown (WILD) samples.
            n_batches_to_process (int, optional): Number of batches to process from each DataLoader. Processes all if None.

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): Combined feature matrix from both ID and WILD samples.
                - y (np.ndarray): Labels indicating ID (0) and WILD (2) samples.
        """
        print("Computing features...")

        if self.X is not None:
            return self.X, self.y

        features = []
        labels = []

        with torch.no_grad():
            # Process ID DataLoader
            for batch_idx, batch in tqdm.tqdm(
                enumerate(self.id_loader),
                total=self.n_batches_to_process,
                desc="Processing ID batches",
            ):
                if (
                    self.n_batches_to_process is not None
                    and batch_idx >= self.n_batches_to_process
                ):
                    break

                aug_input = batch["image"].to(self.device) / 3000
                _ = self.forward(aug_input)

                id_features = self.hooked_features[-1]
                self.hooked_features.clear()
                id_pooled = F.adaptive_max_pool2d(id_features, (1, 1)).view(
                    id_features.size(0), -1
                )

                features.append(id_pooled)
                labels.append(np.zeros(id_pooled.size(0)))  # Label ID as 0
                self.id_coords.append(batch["coords"])

            del batch, batch_idx, id_features, id_pooled

            # Process Unknown DataLoader
            for batch_idx, batch in tqdm.tqdm(
                enumerate(self.wild_loader),
                total=self.n_batches_to_process,
                desc="Processing WILD batches",
            ):
                if (
                    self.n_batches_to_process is not None
                    and batch_idx >= self.n_batches_to_process
                ):
                    break

                aug_wild = batch["image"].to(self.device) / 3000

                _ = self.forward(aug_wild)

                wild_features = self.hooked_features[-1]
                self.hooked_features.clear()
                ood_pooled = F.adaptive_max_pool2d(wild_features, (1, 1)).view(
                    wild_features.size(0), -1
                )

                features.append(ood_pooled)
                labels.append(np.full(ood_pooled.size(0), 2))  # Label WILD as 2
                self.wild_coords.append(batch["coords"])

        # Combine all features and labels
        X = torch.cat(features, dim=0).cpu().numpy()
        y = np.concatenate(labels, axis=0)

        self.X = X
        self.y = y

        return self.X, self.y

    def feature_space_clustering(self, X, y):
        """
        Perform clustering on the feature space and assign labels for OOD detection.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): Initial labels (0 for ID, 2 for WILD).

        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): The input feature matrix.
                - y_clustered (np.ndarray): Labels indicating ID (0) and WILD (1) samples after clustering.
        """
        print("Clustering feature space...")

        if self.y_clustered is not None:
            return self.X, self.y_clustered

        if self.use_optuna:
            # Fine-tune clustering parameters k and M using Optuna
            best_params = fine_tune_clustering_params(X, y)
            self.num_clusters = best_params["k"]
            self.M = best_params["M"]
            # Log the best trial's metrics
            self.best_clustering_metrics = best_params["metrics"]
            print(
                "Best clustering parameters found using Optuna:",
                self.best_clustering_metrics,
            )
        else:
            # Use provided k and M values without tuning
            print(f"Using provided values: k={self.num_clusters}, M={self.M}")

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X)
        id_fraction_per_cluster = [
            np.mean(y[cluster_labels == cluster] == 0)
            for cluster in range(self.num_clusters)
        ]
        ood_cluster_indices = [
            i for i, fraction in enumerate(id_fraction_per_cluster) if fraction < self.M
        ]
        self.y_clustered = np.ones_like(y, dtype=np.int8)
        for cluster_idx in range(self.num_clusters):
            if cluster_idx not in ood_cluster_indices:
                self.y_clustered[cluster_labels == cluster_idx] = 0
        return self.y_clustered

    def save_classifier(self):
        """Save the trained OOD classifier to a file."""
        if self.ood_classifier is not None:
            joblib.dump(self.ood_classifier, self.classifier_save_path)
            print(f"OOD classifier saved to {self.classifier_save_path}")
        else:
            print("No OOD classifier to save.")

    def load_classifier(self):
        """Load the OOD classifier from a file if it exists."""
        try:
            self.ood_classifier = joblib.load(self.classifier_save_path)
            print(f"OOD classifier loaded from {self.classifier_save_path}")
        except FileNotFoundError:
            print(
                f"No saved OOD classifier found at {self.classifier_save_path}. A new classifier will be trained."
            )

    def g_classification(self, X, y, save_classifier=True):
        """
        Train a classifier on the clustered features and evaluate its performance.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The labels (0 for Surrogate ID, 1 for Surrogate OOD).
            save_classifier (bool): Whether to save the trained classifier.

        Returns:
            dict: A dictionary containing evaluation metrics such as accuracy, ROC AUC, FPR95, and a classification report.

        Raises:
            ValueError: If the validation set contains only one class.
        """

        if self.ood_classifier is not None:
            print("Using existing OOD classifier...")
            return self.ood_classifier_metrics

        print("Training OOD classifier...")

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Train classifier
        clf = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=self.random_state
        )

        model = make_pipeline(StandardScaler(), clf)
        model.fit(X_train, y_train)

        self.ood_classifier = model

        # Evaluate classifier
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        clsf_report_dict = classification_report(
            y_val, y_pred, digits=3, output_dict=True
        )

        # Compute AUC and FPR95
        if len(np.unique(y_val)) == 1:
            raise ValueError(
                "Error in calculating AUC: Only one class present in the validation set."
            )

        y_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)

        # Compute FPR95%
        idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
        fpr95 = fpr[idx] if idx != -1 else np.nan

        # Log results
        self.ood_classifier_metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "fpr95": fpr95,
            "classification_report": clsf_report_dict,
            "y_proba": y_proba,
        }

        if save_classifier:
            self.save_classifier()

        return self.ood_classifier_metrics

    def f_g_prediction(self, new_tensor, upsample=False):
        """
        Predict the model's output and classify the samples as ID (0) or OOD (1).

        Args:
            new_tensor (torch.Tensor): A tensor representing the new samples to classify.

        Returns:
            tuple: A tuple containing:
                - predictions (torch.Tensor): The model's predictions for the new samples.
                - ood_labels (np.ndarray): labels (0 for ID, 1 for OOD).
        """
        # Ensure the OOD classifier is loaded or trained
        if self.ood_classifier is None:
            if self.X is None or self.y_clustered is None:
                X, y = self.compute_features(
                    self.id_loader, self.unknown_loader, self.n_batches_to_process
                )
                y_clustered = self.feature_space_clustering(X, y)
                self.ood_classification(X, y_clustered)

        # Pass the new tensor through the model to get predictions and extract features
        with torch.no_grad():
            self.hooked_features.clear()
            if upsample:
                new_tensor = self.up_sample(new_tensor).float()
            new_tensor = new_tensor / 3000
            f_pred = self.forward(new_tensor)
            new_features = self.hooked_features[-1]
            if upsample:
                f_pred = self.down_sample(f_pred.float())

        # Downsample features
        new_pooled = (
            F.adaptive_max_pool2d(new_features, (1, 1))
            .view(new_features.size(0), -1)
            .cpu()
            .numpy()
        )

        # Predict OOD status using the trained classifier
        g_pred_proba = self.ood_classifier.predict_proba(new_pooled)[:, 1]

        return f_pred, g_pred_proba


def calculate_average_entropy(cluster_labels, true_labels, num_clusters):
    cluster_entropy = []
    for cluster_idx in range(num_clusters):
        # True labels for samples in the current cluster
        cluster_true_labels = true_labels[cluster_labels == cluster_idx]
        if len(cluster_true_labels) == 0:
            continue
        # Calculate the proportion of ID and OOD samples in the cluster
        p_id = np.mean(cluster_true_labels == 0)
        p_ood = np.mean(cluster_true_labels == 2)
        # Calculate the entropy for the cluster
        entropy_value = entropy([p_id, p_ood])
        cluster_entropy.append(entropy_value)
    return np.mean(cluster_entropy) if cluster_entropy else 0


def fine_tune_clustering_params(X_search, y_search):
    """
    Fine-tune the clustering parameters k and M using Optuna.

    Args:
        X_search (np.ndarray): The feature matrix for the search.
        y_search (np.ndarray): The labels for the search.

    Returns:
        dict: A dictionary containing the best parameters and metrics.
    """

    def objective(trial):
        # Dynamically calculate X and Y based on the length of X_search
        len_X_search = len(X_search)
        X = int(0.1 * len_X_search)  # 10% of the data
        Y = int(0.4 * len_X_search)  # 40% of the data

        # Suggest values for the hyperparameters within the constrained range for k
        k = trial.suggest_int("k", X, Y)
        M = trial.suggest_float("M", 0.01, 0.3)

        # Fit KMeans clustering
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
        cluster_labels = kmeans.fit_predict(X_search)
        id_fraction_per_cluster = [
            np.mean(y_search[cluster_labels == cluster] == 0) for cluster in range(k)
        ]

        ood_cluster_indices = [
            i for i, fraction in enumerate(id_fraction_per_cluster) if fraction < M
        ]

        y_clusters = np.ones_like(y_search, dtype=np.int8)
        for cluster_idx in range(k):
            if cluster_idx not in ood_cluster_indices:
                y_clusters[cluster_labels == cluster_idx] = 0

        correct_id = np.sum((y_clusters == 0) & (y_search == 0))
        incorrect_id = np.sum((y_clusters == 0) & (y_search == 1))
        correct_id_proportion = correct_id / (correct_id + incorrect_id)
        incorrect_id_proportion = incorrect_id / (correct_id + incorrect_id)

        # Train a classifier on the clustered features
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model = make_pipeline(StandardScaler(), clf)
        model.fit(X_search, y_clusters)

        # Evaluate the classifier
        y_pred = model.predict(X_search)
        accuracy = accuracy_score(y_clusters, y_pred)

        y_proba = model.predict_proba(X_search)[:, 1]
        fpr, tpr, _ = roc_curve(y_clusters, y_proba)
        roc_auc = auc(fpr, tpr)

        # Calculate average entropy
        average_entropy = calculate_average_entropy(cluster_labels, y_search, k)

        # FPR at 95% TPR
        fpr95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
        fpr95 = fpr[fpr95_idx] if fpr95_idx != -1 else np.nan

        # Store all metrics in trial user attributes
        result = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "fpr95": fpr95,
            "correct_id_proportion": correct_id_proportion,
            "incorrect_id_proportion": incorrect_id_proportion,
            "average_entropy": average_entropy,
        }

        for key, value in result.items():
            trial.set_user_attr(key, value)
        # Composite objective: minimize entropy and incorrect_id_proportion, maximize correct_id_proportion
        return average_entropy + incorrect_id_proportion - correct_id_proportion

    # Set up the study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # Output the best parameters found
    best_trial = study.best_trial
    best_params = {
        "k": best_trial.params["k"],
        "M": best_trial.params["M"],
        "metrics": best_trial.user_attrs,
    }
    return best_params
