# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OOD Classifier."""

import gc
import json
import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import set_seed


def save_cfg_as_yaml(cfg, filename):
    yaml = YAML()
    yaml.default_flow_style = False
    with open(filename, "w") as file:
        yaml.dump(cfg, file)


def plot_confusion_matrix(cm, title, save_path=None, show_plot=False):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, title, save_path=None, show_plot=False):
    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_precision_recall_curve(
    precision, recall, title, save_path=None, show_plot=False
):
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def train_evaluate_log_ood_classifier(
    X,
    y,
    classifier_data_path,
    property_lengths,
    test_size,
    n_estimators,
    random_state,
    folder_name,
    create_plots,
    save_plots,
    save_model=False,
    clf_name="RandomForestClassifier",
    **kwargs,
):
    set_seed(random_state)

    if classifier_data_path is not None and folder_name is not None:
        exp_folder = os.path.join(classifier_data_path, folder_name)
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        print(f"Experiment folder created: {exp_folder}")

    # Split data: training on x_train y_train evaluating on y_test and x_test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Check class distribution
    class_counts = Counter(y_train)
    min(class_counts.values())

    if clf_name == "RandomForestClassifier":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )

    elif clf_name == "LogisticRegressionCV":
        clf = LogisticRegressionCV(cv=2, random_state=random_state)
    elif clf_name == "LogisticRegression":
        clf = LogisticRegression(random_state=random_state)
    else:
        raise ValueError("Invalid classifier name.")

    # Train classifier
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    clsf_report_dict = classification_report(y_val, y_pred, digits=3, output_dict=True)

    # Compute AUC
    y_proba = (
        model.predict_proba(X_val)[:, 1]
        if hasattr(clf, "predict_proba")
        else model.decision_function(X_val)
    )
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)

    # Compute FPR95%
    idx = np.where(tpr >= 0.95)[0][0]
    fpr95 = fpr[idx]

    # Log results
    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "fpr95": fpr95,
        "classification_report": clsf_report_dict,
    }

    clf_name = clf.__class__.__name__

    if create_plots or save_plots:
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plot_confusion_matrix(
            cm,
            title=f"{clf_name} Confusion Matrix",
            save_path=(
                os.path.join(exp_folder, f"{clf_name}_confusion_matrix.png")
                if save_plots
                else None
            ),
            show_plot=create_plots,
        )

        # ROC Curve and AUC
        plot_roc_curve(
            fpr,
            tpr,
            roc_auc,
            title=f"{clf_name} Receiver Operating Characteristic",
            save_path=(
                os.path.join(exp_folder, f"{clf_name}_roc_curve.png")
                if save_plots
                else None
            ),
            show_plot=create_plots,
        )

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_val, y_proba)
        plot_precision_recall_curve(
            precision,
            recall,
            title=f"{clf_name} Precision-Recall Curve",
            save_path=(
                os.path.join(exp_folder, f"{clf_name}_precision_recall_curve.png")
                if save_plots
                else None
            ),
            show_plot=create_plots,
        )

    if classifier_data_path is not None and folder_name is not None:
        # Generate a unique filename for the experiment based on arguments
        experiment_details = f"testsize-{test_size}_randomstate-{random_state}"
        filename_prefix = os.path.join(
            exp_folder, f"{folder_name}_{experiment_details}"
        )

        if save_model:
            # Save the model
            model_save_path = f"{filename_prefix}_model.pkl"
            joblib.dump(model, model_save_path)

        # Save metrics and log results to a file
        log_save_path = f"{filename_prefix}_log.json"
        with open(log_save_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Model and log saved to: {exp_folder}")

    # Clean up
    gc.collect()
    return metrics
