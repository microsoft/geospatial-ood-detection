# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing utilities for xBD and EuroSAT datasets."""

import gc
import random
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torchgeo.trainers import ClassificationTask

sys.path.append("..")
from .eurosat_datamodule import (EuroSATClassHoldOutAug,
                                 EuroSATSpatialDataModuleAug)
from .utils import create_feature_matrix_and_labels


def set_task_model_cfg_dict(config_path, ckpt_path, device, mode):
    """
    Loads the configuration, model, and data module for the EuroSAT classification task.
    Args:
        config_path (str): Path to the configuration file.
        ckpt_path (str): Path to the model checkpoint file.
    Returns:
        tuple: A tuple containing the following elements:
            - task (ClassificationTask): The loaded classification task.
            - model (torch.nn.Module): The loaded and evaluated model.
            - datamodule (EuroSATClassHoldOutAug): The data module for the EuroSAT dataset.
            - train_dataloader (DataLoader): DataLoader for the training dataset.
            - val_dataloader (DataLoader): DataLoader for the validation dataset.
            - test_dataloader (DataLoader): DataLoader for the test dataset.
            - cfg (OmegaConf): The loaded configuration.
    """

    cfg = OmegaConf.load(config_path)
    datetime.now().strftime("%Y%m%d_%H%M%S")

    task = ClassificationTask.load_from_checkpoint(ckpt_path)
    task.freeze()
    model = task.model
    model = model.eval().to(device)
    print("Model and Task are loaded.")

    if mode == "holdout":
        datamodule = EuroSATClassHoldOutAug(
            class_name=cfg.data.class_name,
            root=cfg.data.root,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            exclude_class=cfg.data.class_name,
            download=cfg.data.download,
            checksum=cfg.data.checksum,
        )
    elif mode == "spatialsplit":
        datamodule = EuroSATSpatialDataModuleAug(
            root=cfg.data.root,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sampler=None,
        )

    datamodule.setup(stage="fit")
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()
    print("DataModule setup complete.")

    return (
        task,
        model,
        datamodule,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        cfg,
    )


def get_X_y_arrays(
    config_path,
    ckpt_path,
    layer,
    downsample_method,
    getitem_keys,
    device,
    n_batches_to_process,
    mode,
    collect_aug_input=False,
    verbose=False,
):
    """
    Extracts feature matrices and labels from a given model and datamodule configuration.
    Args:
        config_path (str): Path to the configuration file.
        ckpt_path (str): Path to the model checkpoint file.
        layer (str): The layer name from which features are to be extracted.
        downsample_method (str): Method to downsample the features.
        getitem_keys (list): List of keys to extract from the dataloader.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        n_batches_to_process (int): Number of batches to process.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The feature matrix.
            - y (numpy.ndarray): The labels.
    """

    (
        task,
        model,
        datamodule,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        cfg_dict,
    ) = set_task_model_cfg_dict(config_path, ckpt_path, device, mode)
    print("Model -- Task -- Dataloaders are loaded.")
    print(
        "len(train_dataloader), len(val_dataloader), len(test_dataloader): ",
        len(train_dataloader),
        len(val_dataloader),
        len(test_dataloader),
    )

    X, y, test_property_lengths, train_images, test_images = (
        create_feature_matrix_and_labels(
            model=model,
            dm=datamodule,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            layer_names=layer,
            device=device,
            getitem_keys=getitem_keys,
            n_batches_to_process=n_batches_to_process,
            downsample_method=downsample_method,
            verbose=verbose,
            collect_aug_input=collect_aug_input,
        )
    )
    return (
        X,
        y,
        model,
        datamodule,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        cfg_dict,
        train_images,
        test_images,
    )


def run_g_experiment(
    X, y, split_seed, test_size, n_estimators, fixed_classifier_seed, clf=None
):
    # Split the data into training and validation sets
    X_train_cluster, X_val_baseline, y_train_cluster, y_val_baseline = train_test_split(
        X, y, test_size=test_size, random_state=split_seed
    )

    if clf is None:
        clf = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=fixed_classifier_seed
        )

    # Create a pipeline with scaling and the classifier
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train_cluster, y_train_cluster)

    # Evaluate classifier
    y_pred = model.predict(X_val_baseline)
    baseline_accuracy = accuracy_score(y_val_baseline, y_pred)

    # Compute AUC
    y_proba = (
        model.predict_proba(X_val_baseline)[:, 1]
        if hasattr(clf, "predict_proba")
        else model.decision_function(X_val_baseline)
    )

    fpr, tpr, thresholds = roc_curve(y_val_baseline, y_proba)
    baseline_roc_auc = auc(fpr, tpr)

    # Compute FPR95%
    idx = np.where(tpr >= 0.95)[0][0]
    baseline_fpr95 = fpr[idx]

    # Calculate number of OOD and ID samples
    actual_ood_samples = np.sum(y_train_cluster == 1)
    actual_id_samples = np.sum(y_train_cluster == 0)

    results_dict = {
        "baseline_accuracy": baseline_accuracy,
        "baseline_roc_auc": baseline_roc_auc,
        "baseline_fpr95": baseline_fpr95,
        "actual_ood_samples": actual_ood_samples,
        "actual_id_samples": actual_id_samples,
    }

    gc.collect()

    return results_dict


def run_multiple_experiments_g(
    X,
    y,
    test_size,
    n_estimators,
    N=10,
    fixed_classifier_seed=31,
    random_seed=False,
    clf=None,
):
    results_list = []
    for i in range(N):
        if random_seed:
            split_seed = np.random.randint(1, 10000)
        else:
            split_seed = fixed_classifier_seed
        print(f"random_state for split_seed: {split_seed}")

        results = run_g_experiment(
            X=X,
            y=y,
            split_seed=split_seed,
            test_size=test_size,
            n_estimators=n_estimators,
            fixed_classifier_seed=fixed_classifier_seed,
            clf=clf,
        )

        # Add the iteration number to results
        results["iteration"] = i + 1
        results_list.append(results)

    # Convert the list of results into a DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df


def calculate_mean_std(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Standard deviation
    return mean, std


# Function to calculate the confidence interval
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    stderr = stats.sem(data)  # Standard error of the mean
    h = stderr * stats.t.ppf((1 + confidence) / 2.0, n - 1)  # Margin of error
    return mean, h  # Returning mean and margin of error


# Calculate confidence intervals for the specified columns
def calculate_confidence_intervals(results_df, columns_of_interest):
    """
    Calculate confidence intervals for the specified columns in the results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results.
        columns_of_interest (list): List of column names for which to calculate confidence intervals.

    Returns:
        dict: A dictionary with column names as keys and tuples of (mean, margin_of_error) as values.
    """
    confidence_intervals = {}
    for column in columns_of_interest:
        mean, margin_of_error = confidence_interval(results_df[column])
        confidence_intervals[column] = (mean, margin_of_error)
    return confidence_intervals


def calculate_average_entropy(cluster_labels, true_labels, num_clusters):
    cluster_entropy = []
    for cluster_idx in range(num_clusters):
        # True labels for samples in the current cluster
        cluster_true_labels = true_labels[cluster_labels == cluster_idx]
        if len(cluster_true_labels) == 0:
            continue
        # Calculate the proportion of ID and OOD samples in the cluster
        p_id = np.mean(cluster_true_labels == 0)
        p_ood = np.mean(cluster_true_labels == 1)
        # Calculate the entropy for the cluster
        entropy_value = entropy([p_id, p_ood])
        cluster_entropy.append(entropy_value)
    return np.mean(cluster_entropy) if cluster_entropy else 0


# Define the objective function to optimize
def objective(
    trial,
    X,
    y,
    test_size,
    min_cluster,
    max_cluster_ratio,
    min_fraction,
    max_fraction,
    n_estimators,
    fixed_seed,
    clf=None,
):
    # Split the data into training and validation sets
    X_train_cluster, X_val_baseline, y_train_cluster, y_val_baseline = train_test_split(
        X, y, test_size=test_size, random_state=fixed_seed
    )

    min_cluster = min_cluster
    max_cluster = int(max_cluster_ratio * len(X_train_cluster))

    min_fraction = min_fraction
    max_fraction = max_fraction

    # Suggest values for the hyperparameters within the constrained range for k and M
    k = trial.suggest_int("k", min_cluster, max_cluster)
    M = trial.suggest_float("M", min_fraction, max_fraction)

    # Fit KMeans clustering
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=fixed_seed)
    cluster_labels = kmeans.fit_predict(X_train_cluster)

    id_fraction_per_cluster = [
        np.mean(y_train_cluster[cluster_labels == cluster] == 0) for cluster in range(k)
    ]

    ood_cluster_indices = [
        i for i, fraction in enumerate(id_fraction_per_cluster) if fraction < M
    ]

    y_clusters = np.ones_like(y_train_cluster, dtype=np.int8)
    for cluster_idx in range(k):
        if cluster_idx not in ood_cluster_indices:
            y_clusters[cluster_labels == cluster_idx] = 0

    # Check for unique labels in y_clusters
    if len(np.unique(y_clusters)) < 2:
        # Return a large penalty to indicate failure or invalid trial
        return float("inf")

    correct_id = np.sum((y_train_cluster == 0) & (y_train_cluster == 0))
    incorrect_id = np.sum((y_train_cluster == 0) & (y_train_cluster == 1))
    actual_id_samples = np.sum(y_train_cluster == 0)
    correct_id_proportion = correct_id / actual_id_samples
    incorrect_id_proportion = (
        incorrect_id / actual_id_samples if actual_id_samples > 0 else 0
    )

    if clf is None:
        # Fit the classifier
        clf = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=fixed_seed
        )

    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train_cluster, y_clusters)

    y_pred = model.predict(X_val_baseline)
    accuracy = accuracy_score(y_val_baseline, y_pred)

    # Try-except block to handle potential errors
    try:
        y_proba = model.predict_proba(X_val_baseline)[:, 1]
    except IndexError:
        return float(
            "inf"
        )  # Handle the case where predict_proba has only one class (likely to be 0)

    fpr, tpr, _ = roc_curve(y_val_baseline, y_proba)
    roc_auc = auc(fpr, tpr)

    # Calculate average entropy
    average_entropy = calculate_average_entropy(cluster_labels, y_train_cluster, k)

    # FPR at 95% TPR
    fpr95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
    fpr95 = fpr[fpr95_idx] if fpr95_idx != -1 else np.nan

    # Store all metrics in trial user attributes
    result = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "fpr95": fpr95,
        "actual_ood_samples": np.sum(
            y_train_cluster == 1
        ),  # Ensure this is defined correctly
        "surrogate_ood": np.sum(y_clusters == 1),
        "correct_ood": np.sum((y_clusters == 1) & (y_train_cluster == 1)),
        "incorrect_ood": np.sum((y_clusters == 1) & (y_train_cluster == 0)),
        "actual_id_samples": actual_id_samples,
        "surrogate_id": np.sum(y_clusters == 0),
        "correct_id": correct_id,
        "incorrect_id": incorrect_id,
        "correct_id_proportion": correct_id_proportion,
        "incorrect_id_proportion": incorrect_id_proportion,
        "average_entropy": average_entropy,
    }

    for key, value in result.items():
        trial.set_user_attr(key, value)

    # Composite objective: minimize entropy and incorrect_id_proportion, maximize correct_id_proportion
    return average_entropy + incorrect_id_proportion - correct_id_proportion


def run_optuna_study(
    X,
    y,
    n_optuna_trials,
    test_size,
    min_cluster,
    max_cluster_ratio,
    min_fraction,
    max_fraction,
    n_estimators,
    fixed_seed,
):
    results = []

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=fixed_seed),
    )
    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
            test_size,
            min_cluster,
            max_cluster_ratio,
            min_fraction,
            max_fraction,
            n_estimators,
            fixed_seed,
        ),
        n_trials=n_optuna_trials,
    )

    best_trial = study.best_trial

    results.append(
        {
            "trial_number": best_trial.number,
            "k": best_trial.params.get("k", None),
            "M": best_trial.params.get("M", None),
            "correct_id_proportion": best_trial.user_attrs.get(
                "correct_id_proportion", None
            ),
            "incorrect_id_proportion": best_trial.user_attrs.get(
                "incorrect_id_proportion", None
            ),
            "average_entropy": best_trial.user_attrs.get("average_entropy", None),
            "accuracy": best_trial.user_attrs.get("accuracy", None),
            "fpr95": best_trial.user_attrs.get("fpr95", None),
            "roc_auc": best_trial.user_attrs.get("roc_auc", None),
            "trial_value": best_trial.value,
        }
    )

    results_df = pd.DataFrame(results)
    return results_df


def run_g_hat_experiment(
    X, y, test_size, k, M, split_seed, fixed_seed, iteration_num, clf=None
):
    # Split the data
    X_train_cluster, X_val_baseline, y_train_cluster, y_val_baseline = train_test_split(
        X, y, test_size=test_size, random_state=split_seed
    )

    # Get the indices of train and baseline data
    train_indices = np.where(np.isin(X, X_train_cluster))[0]
    baseline_indices = np.where(np.isin(X, X_val_baseline))[0]

    # Fit KMeans clustering
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=fixed_seed)
    cluster_labels = kmeans.fit_predict(X_train_cluster)

    # Calculate ID fraction for each cluster
    id_fraction_per_cluster = [
        np.mean(y_train_cluster[cluster_labels == cluster] == 0) for cluster in range(k)
    ]

    # Identify OOD clusters
    ood_cluster_indices = [
        i for i, fraction in enumerate(id_fraction_per_cluster) if fraction < M
    ]

    # Assign clusters as ID (0) or OOD (1)
    y_clusters = np.ones_like(y_train_cluster, dtype=np.int8)
    for cluster_idx in range(k):
        if cluster_idx not in ood_cluster_indices:
            y_clusters[cluster_labels == cluster_idx] = 0

    # Use the provided classifier or default to RandomForestClassifier
    if clf is None:
        clf = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=fixed_seed
        )

    # Fit the classifier pipeline
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train_cluster, y_clusters)

    # Evaluate the model
    y_pred = model.predict(X_val_baseline)
    accuracy = accuracy_score(y_val_baseline, y_pred)

    # Calculate ROC AUC and FPR at 95% TPR
    y_proba = model.predict_proba(X_val_baseline)[:, 1]
    fpr, tpr, _ = roc_curve(y_val_baseline, y_proba)
    roc_auc = auc(fpr, tpr)

    fpr95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
    fpr95 = fpr[fpr95_idx] if fpr95_idx != -1 else np.nan

    # Return metrics and data
    metrics = {
        "iteration": iteration_num,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "fpr95": fpr95,
    }

    return (
        metrics,
        X_train_cluster,
        y_train_cluster,
        y_clusters,
        train_indices,
        baseline_indices,
    )


def run_multiple_experiments_g_hat(X, y, test_size, k, M, N, fixed_seed=31, clf=None):
    """
    Runs N iterations of the g_hat_experiment and stores the results.
    Returns:
    results_list : list
        A list of result dictionaries for each iteration.
    """
    results_list = []

    for i in range(N):
        # Generate a new random split_seed for each iteration
        split_seed = np.random.randint(1, 10000)
        print(f"random_state for split_seed in iteration {i + 1}:", split_seed)

        # Call the single experiment function
        result, X_train_cluster, y_train_cluster, y_clusters = run_g_hat_experiment(
            X, y, test_size, k, M, split_seed, fixed_seed, i + 1, clf
        )
        results_list.append(result)
    return results_list


def benchmark_classifiers(X, y, test_size, k, M, classifiers, fixed_seed):
    """
    Benchmarks different classifiers on the given dataset, measuring performance and walltime.

    Parameters:
    X : array-like
        The feature set.
    y : array-like
        The labels.
    test_size : float
        Proportion of the dataset to include in the test split.
    k : int
        Number of clusters for KMeans.
    M : float
        Threshold for marking a cluster as OOD.
    classifiers : dict
        Dictionary of classifiers to benchmark.

    Returns:
    results_df : pandas.DataFrame
        A DataFrame containing the benchmark results.
    """

    # Data split
    X_train_cluster, X_val_baseline, y_train_cluster, y_val_baseline = train_test_split(
        X, y, test_size=test_size, random_state=fixed_seed
    )

    # Compute sample weights for models that don't have `class_weight` parameter
    compute_sample_weight("balanced", y_train_cluster)

    # Benchmark KMeans clustering walltime
    start_time_cluster = time.time()

    # Fit KMeans clustering
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=fixed_seed)
    cluster_labels = kmeans.fit_predict(X_train_cluster)

    cluster_end_time = time.time()
    clustering_time = cluster_end_time - start_time_cluster

    # Calculate ID fraction per cluster
    id_fraction_per_cluster = [
        np.mean(y_train_cluster[cluster_labels == cluster] == 0) for cluster in range(k)
    ]

    ood_cluster_indices = [
        i for i, fraction in enumerate(id_fraction_per_cluster) if fraction < M
    ]

    y_clusters = np.ones_like(y_train_cluster, dtype=np.int8)
    for cluster_idx in range(k):
        if cluster_idx not in ood_cluster_indices:
            y_clusters[cluster_labels == cluster_idx] = 0

    # Store results
    benchmark_results = []

    # Loop through classifiers and measure walltimes for training and prediction
    for clf_name, clf in classifiers.items():
        model = make_pipeline(StandardScaler(), clf)

        # Benchmark classifier training walltime
        start_time_clsf = time.time()

        # Fit model (train)
        model.fit(X_train_cluster, y_clusters)

        end_time_clsf = time.time()
        classifier_time = end_time_clsf - start_time_clsf

        start_time_clsf_pred = time.time()
        # Make predictions
        y_pred = model.predict(X_val_baseline)

        end_time_clsf_pred = time.time()
        classifier_pred_time = end_time_clsf_pred - start_time_clsf_pred

        accuracy = accuracy_score(y_val_baseline, y_pred)

        # Predict probabilities for AUROC and FPR95
        try:
            y_proba = model.predict_proba(X_val_baseline)[:, 1]
        except IndexError:
            y_proba = np.zeros_like(
                y_val_baseline
            )  # Handle case where only one class is predicted

        fpr, tpr, _ = roc_curve(y_val_baseline, y_proba)
        roc_auc = auc(fpr, tpr)

        fpr95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
        fpr95 = fpr[fpr95_idx] if fpr95_idx != -1 else np.nan

        # Calculate time per sample
        classifier_time_per_sample = classifier_time / len(
            X_train_cluster
        )  # Fit time per sample
        classifier_pred_time_per_sample = classifier_pred_time / len(
            X_val_baseline
        )  # Prediction time per sample

        # Append the results
        benchmark_results.append(
            {
                "Classifier": clf_name,
                "Accuracy": accuracy,
                "ROC_AUC": roc_auc,
                "FPR95": fpr95,
                "Number of Train Samples": len(X_train_cluster),
                "Number of Val Samples": len(X_val_baseline),
                "Cluster Fit Time (s)": clustering_time,
                "Classifier Fit Time (s)": classifier_time,
                "Classifier Fit Time (s/sample)": classifier_time_per_sample,
                "Classifier Pred Time (s)": classifier_pred_time,
                "Classifier Pred Time (s/sample)": classifier_pred_time_per_sample,
            }
        )

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(benchmark_results)
    return results_df


def benchmark_clustering_methods(X, y, test_size, k, M, clustering_methods, classifier):
    """
    Benchmarks different clustering methods followed by a classifier on the given dataset.

    Parameters:
    X : array-like
        The feature set.
    y : array-like
        The labels.
    test_size : float
        Proportion of the dataset to include in the test split.
    k : int
        Number of clusters for KMeans (if used).
    M : float
        Threshold for marking a cluster as OOD.
    clustering_methods : dict
        Dictionary of clustering methods to benchmark.
    classifier : estimator
        A classifier model to benchmark.

    Returns:
    results_df : pandas.DataFrame
        A DataFrame containing the benchmark results for each clustering method.
    """

    # Data split
    X_train_cluster, X_val_baseline, y_train_cluster, y_val_baseline = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Log the number of samples
    n_samples = len(X_train_cluster)

    # Compute sample weights for models that don't have `class_weight` parameter
    compute_sample_weight("balanced", y_train_cluster)

    # Store results
    benchmark_results = []

    # Loop through clustering methods and log walltimes
    for cluster_name, clustering in clustering_methods.items():
        # Log clustering walltime
        start_time = time.time()

        # Perform clustering
        if cluster_name.startswith("KMeans"):
            cluster_labels = clustering.fit_predict(X_train_cluster)
        else:  # For DBSCAN
            cluster_labels = clustering.fit_predict(X_train_cluster)

        end_time = time.time()
        clustering_time = end_time - start_time

        # Calculate OOD cluster indices based on fraction of ID samples
        id_fraction_per_cluster = [
            np.mean(y_train_cluster[cluster_labels == cluster] == 0)
            for cluster in np.unique(cluster_labels)
            if cluster != -1  # Exclude DBSCAN's noise cluster
        ]

        ood_cluster_indices = [
            i for i, fraction in enumerate(id_fraction_per_cluster) if fraction < M
        ]

        # Assign OOD and ID labels to clusters
        y_clusters = np.ones_like(y_train_cluster, dtype=np.int8)
        for cluster_idx in np.unique(cluster_labels):
            if (
                cluster_idx != -1 and cluster_idx not in ood_cluster_indices
            ):  # Exclude noise for DBSCAN
                y_clusters[cluster_labels == cluster_idx] = 0

        # Log classifier training walltime
        start_time = time.time()

        # Fit the classifier
        model = make_pipeline(StandardScaler(), classifier)
        model.fit(X_train_cluster, y_clusters)

        end_time = time.time()
        classifier_time = end_time - start_time

        # Predict on the validation set
        y_pred = model.predict(X_val_baseline)
        accuracy = accuracy_score(y_val_baseline, y_pred)

        # Predict probabilities for AUROC and FPR95
        try:
            y_proba = model.predict_proba(X_val_baseline)[:, 1]
        except IndexError:
            y_proba = np.zeros_like(
                y_val_baseline
            )  # Handle case where only one class is predicted

        fpr, tpr, _ = roc_curve(y_val_baseline, y_proba)
        roc_auc = auc(fpr, tpr)

        fpr95_idx = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
        fpr95 = fpr[fpr95_idx] if fpr95_idx != -1 else np.nan

        # Append the results
        benchmark_results.append(
            {
                "Clustering": cluster_name,
                "Accuracy": accuracy,
                "ROC_AUC": roc_auc,
                "FPR95": fpr95,
                "Clustering Time (s)": clustering_time,
                "Classifier Training Time (s)": classifier_time,
                "Number of Samples": n_samples,
            }
        )

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(benchmark_results)
    return results_df


def benchmark_kmeans_with_varying_k_condidence_g_hat(
    X,
    y,
    M,
    test_size=0.2,
    n_runs=10,
    confidence_level=0.95,
    confidence_intervals_g=None,
    clf=None,
    save_plot=False,
    fname=False,
):
    """
    Benchmarks KMeans clustering with varying values of k, followed by a classifier,
    computes confidence intervals over N runs, and plots results.
    """

    # Split the data into clustering base and fixed validation set
    X_cluster_base, X_val_baseline, y_cluster_base, y_val_baseline = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    # Calculate the maximum value of i such that 2^i <= len(X_cluster_base)
    max_i = int(np.log2(len(X_cluster_base)))
    print("Max i:", max_i)

    # Generate k_values based on powers of 2 until 2^max_i, then add len(X_cluster_base)
    k_values = [2**i for i in range(max_i + 1)]  # +1 to include 2^max_i
    if len(X_cluster_base) not in k_values:
        k_values.append(len(X_cluster_base))

    results = []

    # Loop over the k values
    for k in k_values:
        print("k:", k)

        # Initialize lists to store metrics for each run
        accuracy_list = []
        fpr95_list = []
        roc_auc_list = []

        for run in range(n_runs):
            # Use a different random state for each run
            random_state_run = 42 + run

            # Use the same training data across runs
            X_train_cluster = X_cluster_base
            y_train_cluster = y_cluster_base

            # Fit KMeans clustering with varying random state
            kmeans = KMeans(
                n_clusters=k, init="k-means++", random_state=random_state_run
            )
            cluster_labels = kmeans.fit_predict(X_train_cluster)

            # Calculate the fraction of ID samples per cluster
            id_fraction_per_cluster = [
                (
                    np.mean(y_train_cluster[cluster_labels == cluster] == 0)
                    if np.any(cluster_labels == cluster)
                    else 0
                )
                for cluster in range(k)
            ]

            # Identify OOD clusters
            ood_cluster_indices = [
                i for i, fraction in enumerate(id_fraction_per_cluster) if fraction < M
            ]

            # Assign OOD or ID labels to the clusters
            y_clusters = np.ones_like(y_train_cluster, dtype=np.int8)
            for cluster_idx in range(k):
                if cluster_idx not in ood_cluster_indices:
                    y_clusters[cluster_labels == cluster_idx] = 0

            # Check if y_clusters has more than one class
            if len(np.unique(y_clusters)) < 2:
                print(
                    f"Skipping k={k} on run {run+1} due to single class in y_clusters."
                )
                continue

            # Fit the classifier with varying random state
            if clf is None:
                clf_run = LogisticRegression(
                    class_weight="balanced", max_iter=500, random_state=random_state_run
                )
            else:
                clf_run = clf.set_params(random_state=random_state_run)

            model = make_pipeline(StandardScaler(), clf_run)
            model.fit(X_train_cluster, y_clusters)

            # Predict on the validation set
            y_pred = model.predict(X_val_baseline)
            accuracy = accuracy_score(y_val_baseline, y_pred)

            # Predict probabilities for AUROC and FPR95
            try:
                y_proba = model.predict_proba(X_val_baseline)[:, 1]
            except IndexError:
                y_proba = np.zeros_like(y_val_baseline, dtype=float)

            fpr, tpr, thresholds = roc_curve(y_val_baseline, y_proba)
            roc_auc = auc(fpr, tpr)

            # Calculate FPR at 95% TPR
            try:
                fpr95_idx = np.where(tpr >= 0.95)[0][0]
                fpr95 = fpr[fpr95_idx]
            except IndexError:
                fpr95 = np.nan

            # Append metrics to the lists
            accuracy_list.append(accuracy)
            fpr95_list.append(fpr95)
            roc_auc_list.append(roc_auc)

        # Calculate mean and confidence intervals for each metric
        metrics = {
            "accuracy": accuracy_list,
            "fpr95": fpr95_list,
            "roc_auc": roc_auc_list,
        }

        stats_dict = {}
        for metric_name, values in metrics.items():
            values = np.array(values)
            mean = np.nanmean(values)
            sem = stats.sem(values, nan_policy="omit")
            h = sem * stats.t.ppf((1 + confidence_level) / 2.0, len(values) - 1)
            ci_lower = mean - h
            ci_upper = mean + h
            stats_dict[f"{metric_name}_mean"] = mean
            stats_dict[f"{metric_name}_ci_lower"] = ci_lower
            stats_dict[f"{metric_name}_ci_upper"] = ci_upper

        # Add k and k_fraction to the stats_dict
        stats_dict["k"] = k
        stats_dict["k_fraction"] = k / len(X_cluster_base)

        # Append the stats_dict to results
        results.append(stats_dict)

    # Convert the results into a DataFrame
    df_results = pd.DataFrame(results)

    # Plotting with shaded confidence intervals
    plt.figure(figsize=(10, 8))

    # Plot Accuracy with confidence intervals
    plt.plot(
        df_results["k_fraction"],
        df_results["accuracy_mean"],
        marker="o",
        label="Accuracy",
        color="blue",
    )
    plt.fill_between(
        df_results["k_fraction"],
        df_results["accuracy_ci_lower"],
        df_results["accuracy_ci_upper"],
        color="blue",
        alpha=0.2,
    )
    if confidence_intervals_g is not None:
        plt.axhline(
            y=confidence_intervals_g["baseline_accuracy"][0],
            color="blue",
            linestyle="--",
            label="Baseline Accuracy",
        )

    # Plot FPR95 with confidence intervals
    plt.plot(
        df_results["k_fraction"],
        df_results["fpr95_mean"],
        marker="s",
        label="FPR95",
        color="orange",
    )
    plt.fill_between(
        df_results["k_fraction"],
        df_results["fpr95_ci_lower"],
        df_results["fpr95_ci_upper"],
        color="orange",
        alpha=0.2,
    )
    if confidence_intervals_g is not None:
        plt.axhline(
            y=confidence_intervals_g["baseline_fpr95"][0],
            color="orange",
            linestyle="--",
            label="Baseline FPR95",
        )

    # Plot AUROC with confidence intervals
    plt.plot(
        df_results["k_fraction"],
        df_results["roc_auc_mean"],
        marker="x",
        label="AUROC",
        color="green",
    )
    plt.fill_between(
        df_results["k_fraction"],
        df_results["roc_auc_ci_lower"],
        df_results["roc_auc_ci_upper"],
        color="green",
        alpha=0.2,
    )
    if confidence_intervals_g is not None:
        plt.axhline(
            y=confidence_intervals_g["baseline_roc_auc"][0],
            color="green",
            linestyle="--",
            label="Baseline AUROC",
        )

    # Customizing the plot
    plt.xlabel("k / len(X_train_cluster) (Fraction of Training Set)")
    plt.ylabel("Metrics")
    plt.title("Metrics vs. Fraction of Clusters (k / len(X_train_cluster))")
    plt.legend()
    plt.grid(True)

    # Save the plot if asked
    if save_plot is True and fname is not None:
        plt.savefig(
            f"./benchmark/{fname}_benchmark_kmeans_with_varying_k_confidence_g_hat.svg"
        )

    plt.show()

    return df_results


def plot_tsne_with_label_changes(
    X_train_cluster, y_train_cluster, y_clusters, class_name, save_plot=False
):
    """
    Generates and saves a t-SNE plot with original and clustered labels, and visualizes label changes.

    Parameters:
    X_train_cluster : array-like
        The feature set used for training the clustering model.
    y_train_cluster : array-like
        The original labels (ID/OOD).
    y_clusters : array-like
        The new clustered labels (reassigned after clustering).
    class_name : str
        The class name to include in the plot title and save file.
    """

    # Perform t-SNE to reduce dimensionality to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_train_cluster)

    # Define a discrete colormap for the first two graphs (Original Labels and Clustered Labels)
    cmap = ListedColormap(["green", "orange"])

    # Create a 1x3 subplot figure
    plt.figure(figsize=(18, 6))

    # Plot t-SNE with original labels (ID/OOD)
    plt.subplot(1, 3, 1)
    scatter_1 = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y_train_cluster, cmap=cmap, alpha=0.7
    )
    plt.title("Original Labels (ID/OOD)")
    plt.colorbar(scatter_1, ticks=[0, 1], label="Label (0 = ID, 1 = OOD)")

    # Plot t-SNE with cluster labels (reassigned labels)
    plt.subplot(1, 3, 2)
    scatter_2 = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y_clusters, cmap=cmap, alpha=0.7
    )
    plt.title(f"Clustered Labels (y_clusters)")
    plt.colorbar(scatter_2, ticks=[0, 1], label="Label (0 = ID, 1 = OOD)")

    # Calculate label changes
    no_change = y_train_cluster == y_clusters  # Labels that stayed the same
    change_0_to_1 = (y_train_cluster == 0) & (y_clusters == 1)  # ID to OOD
    change_1_to_0 = (y_train_cluster == 1) & (y_clusters == 0)  # OOD to ID

    # Create an array to represent the label change types
    label_change_type = np.zeros_like(y_train_cluster, dtype=int)
    label_change_type[change_0_to_1] = 1  # Mark 0 to 1 changes
    label_change_type[change_1_to_0] = 2  # Mark 1 to 0 changes

    # Define a discrete colormap for label changes: gray for no change, red for 0 to 1, blue for 1 to 0
    cmap = ListedColormap(["gray", "red", "blue"])

    # Calculate the proportion of label changes
    changed_labels = y_train_cluster != y_clusters
    change_proportion = sum(changed_labels) / len(y_train_cluster)
    print(f"Proportion of label changes: {change_proportion:.2f}")

    # Plot label change types in t-SNE space with a discrete colormap
    plt.subplot(1, 3, 3)
    scatter_3 = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=label_change_type, cmap=cmap, alpha=0.7
    )
    plt.title(f"Proportion of label changes: {change_proportion:.2f}")
    plt.colorbar(
        scatter_3,
        ticks=[0, 1, 2],
        label="Change Type (0 = No Change, 1 = 0 to 1, 2 = 1 to 0)",
    )

    # Save the plot if asked
    save_plot = True
    if save_plot:
        plt.savefig(f"./benchmark/{class_name}_tsne_clustering.svg")

    plt.show()


def plot_downsample_benchmark(data, class_name, save_dir="benchmark"):
    """
    Plots a comparison of Accuracy, AUROC, and FPR95 for different downsampling methods.

    Parameters:
    - data: Dictionary containing the benchmark data.
    - class_name: Name of the class (used in the plot title and file name).
    - save_dir: Directory where the plot image will be saved (default is "benchmark").

    Returns:
    - None
    """

    # Extract labels and values
    labels = list(data.keys())
    accuracy = [data[k]["baseline_accuracy"] for k in labels]
    auroc = [data[k]["baseline_roc_auc"] for k in labels]
    fpr95 = [data[k]["baseline_fpr95"] for k in labels]

    # Set up the bar positions and width
    x = np.arange(len(labels))
    width = 0.2

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0072B2", "#E69F00", "#D55E00"]

    # Plot bars
    bars1 = ax.bar(x - width, accuracy, width, label="Accuracy", color=colors[0])
    bars2 = ax.bar(x, auroc, width, label="AUROC", color=colors[1])
    bars3 = ax.bar(x + width, fpr95, width, label="FPR95", color=colors[2])

    # Set plot labels and title
    ax.set_xlabel("Downsampling Methods")
    ax.set_ylabel("Scores")
    ax.set_title(f"Comparison of Accuracy, AUROC, and FPR95 for {class_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Customize grid and spines
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, f"downsample_{class_name}.png"))

    plt.show()


import re


def plot_layer_benchmark(data, all_layer_names, class_name, save_dir="benchmark"):
    """
    Plots a comparison of Accuracy, AUROC, and FPR95 for different layers.

    Parameters:
    - data: Dictionary containing the benchmark data.
    - all_layer_names: List of all layer names in the order they should be sorted.
    - class_name: Name of the class (used in the plot title and file name).
    - save_dir: Directory where the plot image will be saved (default is "benchmark").

    Returns:
    - None
    """

    # Extract the layer names from each key
    layers = [re.search(r"\['(.*?)'\]", k).group(1) for k in data.keys()]
    accuracy = [data[k]["baseline_accuracy"] for k in data.keys()]
    roc_auc = [data[k]["baseline_roc_auc"] for k in data.keys()]
    fpr95 = [data[k]["baseline_fpr95"] for k in data.keys()]

    # Set up the bar positions and width
    x = np.arange(len(layers))
    width = 0.25

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0072B2", "#E69F00", "#D55E00"]

    # Plot bars
    bars1 = ax.bar(x - width, accuracy, width, label="Accuracy", color=colors[0])
    bars2 = ax.bar(x, roc_auc, width, label="AUROC", color=colors[1])
    bars3 = ax.bar(x + width, fpr95, width, label="FPR95", color=colors[2])

    # Find the index of each layer in all_layer_names and sort layers based on these indices
    layer_indices = [all_layer_names.index(layer) for layer in layers]
    sorted_layers_with_indices = sorted(zip(layers, layer_indices), key=lambda x: x[1])

    # Extract the sorted layers and their indices
    sorted_layers = [layer for layer, idx in sorted_layers_with_indices]
    sorted_indices = [idx for layer, idx in sorted_layers_with_indices]

    # Create the x-axis labels with both layer name and index
    layer_labels_with_index = [
        f"{layer} ({idx + 1}/{len(all_layer_names)})"
        for layer, idx in zip(sorted_layers, sorted_indices)
    ]

    # Set plot labels and title
    ax.set_xlabel("Layers")
    ax.set_ylabel("Scores")
    ax.set_title(f"Layer-wise Performance: Accuracy, AUROC, and FPR95 for {class_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels_with_index, rotation=45, ha="right")

    # Customize grid and spines
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend()

    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, f"layer_{class_name}.png"))

    plt.show()


import numpy as np
from scipy.stats import ttest_ind_from_stats


def calculate_mean_std(values):
    """
    Calculate the mean and standard deviation of a list of values.

    Parameters:
    - values: List of numerical values.

    Returns:
    - A tuple containing the mean and standard deviation.
    """
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Use ddof=1 for sample standard deviation
    return mean, std


def perform_benchmark_analysis(g_benchmark, g_hat_benchmark, n_samples=10):
    """
    Perform benchmark analysis and t-tests on baseline and g_hat metrics.

    Parameters:
    - g_benchmark: Dictionary containing baseline benchmark metrics.
    - g_hat_benchmark: List of dictionaries containing g_hat benchmark metrics.
    - n_samples: Number of samples (default is 10).

    Returns:
    - A dictionary containing the results of the t-tests.
    """

    # Extract baseline metrics
    baseline_acc = list(g_benchmark["baseline_accuracy"])
    baseline_roc_auc = list(g_benchmark["baseline_roc_auc"])
    baseline_fpr95 = list(g_benchmark["baseline_fpr95"])

    # Extract g_hat benchmark metrics using list comprehensions
    g_hat_acc = [entry["accuracy"] for entry in g_hat_benchmark]
    g_hat_roc_auc = [entry["roc_auc"] for entry in g_hat_benchmark]
    g_hat_fpr95 = [entry["fpr95"] for entry in g_hat_benchmark]

    # Calculate mean and standard deviation for baseline metrics
    g_baseline_mean_std_accuracy = calculate_mean_std(baseline_acc)
    g_baseline_mean_std_roc_auc = calculate_mean_std(baseline_roc_auc)
    g_baseline_mean_std_fpr95 = calculate_mean_std(baseline_fpr95)

    # Print baseline results
    print(
        f"Baseline Accuracy: Mean = {g_baseline_mean_std_accuracy[0]}, Std Dev = {g_baseline_mean_std_accuracy[1]}"
    )
    print(
        f"Baseline ROC AUC: Mean = {g_baseline_mean_std_roc_auc[0]}, Std Dev = {g_baseline_mean_std_roc_auc[1]}"
    )
    print(
        f"Baseline FPR95: Mean = {g_baseline_mean_std_fpr95[0]}, Std Dev = {g_baseline_mean_std_fpr95[1]}"
    )

    # Calculate mean and standard deviation for g_hat metrics
    g_hat_mean_std_accuracy = calculate_mean_std(g_hat_acc)
    g_hat_mean_std_roc_auc = calculate_mean_std(g_hat_roc_auc)
    g_hat_mean_std_fpr95 = calculate_mean_std(g_hat_fpr95)

    # Print g_hat results
    print(
        f"g_hat Accuracy: Mean = {g_hat_mean_std_accuracy[0]}, Std Dev = {g_hat_mean_std_accuracy[1]}"
    )
    print(
        f"g_hat ROC AUC: Mean = {g_hat_mean_std_roc_auc[0]}, Std Dev = {g_hat_mean_std_roc_auc[1]}"
    )
    print(
        f"g_hat FPR95: Mean = {g_hat_mean_std_fpr95[0]}, Std Dev = {g_hat_mean_std_fpr95[1]}"
    )

    # Perform t-tests
    t_tests = {}

    # Accuracy
    t_acc, p_acc = ttest_ind_from_stats(
        g_baseline_mean_std_accuracy[0],
        g_baseline_mean_std_accuracy[1],
        n_samples,
        g_hat_mean_std_accuracy[0],
        g_hat_mean_std_accuracy[1],
        n_samples,
        equal_var=True,
    )
    t_tests["accuracy"] = (t_acc, p_acc)

    # ROC AUC
    t_rocauc, p_rocauc = ttest_ind_from_stats(
        g_baseline_mean_std_roc_auc[0],
        g_baseline_mean_std_roc_auc[1],
        n_samples,
        g_hat_mean_std_roc_auc[0],
        g_hat_mean_std_roc_auc[1],
        n_samples,
        equal_var=True,
    )
    t_tests["roc_auc"] = (t_rocauc, p_rocauc)

    # FPR95
    t_fpr95, p_fpr95 = ttest_ind_from_stats(
        g_baseline_mean_std_fpr95[0],
        g_baseline_mean_std_fpr95[1],
        n_samples,
        g_hat_mean_std_fpr95[0],
        g_hat_mean_std_fpr95[1],
        n_samples,
        equal_var=True,
    )
    t_tests["fpr95"] = (t_fpr95, p_fpr95)

    # Print the results of the t-tests
    print(f"Accuracy: t = {t_acc}, p = {p_acc}")
    print(f"ROC AUC: t = {t_rocauc}, p = {p_rocauc}")
    print(f"FPR95: t = {t_fpr95}, p = {p_fpr95}")

    return t_tests


def pick_random_layers(all_layer_names, n):
    conv_layers = [layer for layer in all_layer_names if "conv" in layer]
    first_conv = conv_layers[0]  # First conv layer
    last_conv = conv_layers[-1]  # Last conv layer
    middle_layers = conv_layers[1:-1]
    random_layers = random.sample(middle_layers, min(n - 2, len(middle_layers)))
    selected_layers = [first_conv] + random_layers + [last_conv]
    return selected_layers


def get_model_config(config_path, device):
    cfg = OmegaConf.load(config_path)
    exp_results_best_path = load_exp_ev_metrics_json(cfg.exp_name)
    task = SemanticSegmentationTask.load_from_checkpoint(
        exp_results_best_path["best_model_path"]
    )

    task.freeze()
    model = task.model
    model = model.eval().to(device)
    print("Exp Results:\n", exp_results_best_path)
    print("Model loaded and set to evaluation mode.")
    return model, cfg


def normalize(x):
    return x / 255.0


def prepare_datamodule(cfg):
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

    cfg.training.num_workers = 0

    datamodule = XView2DataModuleOOD(
        root=cfg.paths.root,
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

    datamodule.setup("fit")
    datamodule.setup("test")

    datamodule_train = datamodule.train_dataloader()
    datamodule_val = datamodule.val_dataloader()
    datamodule_test = datamodule.test_dataloader()

    return datamodule, datamodule_train, datamodule_val, datamodule_test


def evaluate_model(model, dm, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            aug_batch = dm.test_aug(batch)
            labels = batch["mask"]
            aug_input = aug_batch["image"].to(device)
            logits = model(aug_input)
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Stack all predictions and labels into single numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_labels, all_preds
