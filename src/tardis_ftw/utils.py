# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing utilities for processing ID and WILD sets."""

import os

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.stats import skew
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def load_config(config_path="config.yaml"):
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str): Path to the configuration file (default is "config.yaml").

    Returns:
        dict: Configuration settings as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def calculate_metrics(true_masks, predicted_masks):
    """
    Calculate evaluation metrics (accuracy, precision, recall, and F1 score) for masks.

    Args:
        true_masks (array): Ground truth masks.
        predicted_masks (array): Predicted masks.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    # Flatten the masks to calculate metrics
    true_flat = true_masks.flatten()
    pred_flat = predicted_masks.flatten()

    # Calculate metrics
    accuracy = accuracy_score(true_flat, pred_flat)
    precision = precision_score(
        true_flat, pred_flat, average="weighted", zero_division=0
    )
    recall = recall_score(true_flat, pred_flat, average="weighted", zero_division=0)
    f1 = f1_score(true_flat, pred_flat, average="weighted", zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def extract_data(country_results):
    """
    Extract and structure data from country-specific results.

    Args:
        country_results (dict): Dictionary containing results for multiple countries.

    Returns:
        pd.DataFrame: DataFrame with metrics and skewness values per country.
    """
    data = []
    for country, results in country_results.items():
        g_pred_probs_testsetid = results["g_pred_probs_testsetid"]
        metrics = results["metrics"]

        # Calculate skewness
        skew_prob = skew(g_pred_probs_testsetid)

        data.append(
            {
                "country": country,
                "skew_prob": skew_prob,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    df = pd.DataFrame(data)
    return df


def plot_f_and_g_preds_probab(
    window_a, window_b, ood_label, output, filepath=None, filename=None, spacing=0.02
):
    """
    Plot window_a, window_b, and output with the OOD label.

    Args:
        window_a (array): Image data for the first window.
        window_b (array): Image data for the second window.
        ood_label (str): The OOD label to display.
        output (array): Image data for the output.
        filepath (str): Directory to save the plot. If None, displays the plot.
        filename (str): Name of the file to save.
        spacing (float): Space between subplots (default is 0.02).
    """
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 4, 4], wspace=spacing)

    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    axes[0].imshow(window_a)
    axes[0].axis("off")

    axes[1].imshow(window_b)
    axes[1].axis("off")

    axes[2].imshow(output)
    axes[2].axis("off")

    if filepath is not None and filename is not None:
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(
            os.path.join(filepath, filename),
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def percentile_stretch(image, lower_percentile=2, upper_percentile=98):
    """
    Normalize an image using percentile-based clipping.

    Args:
        image (array or Tensor): Input image.
        lower_percentile (int): Lower percentile for clipping (default is 2).
        upper_percentile (int): Upper percentile for clipping (default is 98).

    Returns:
        array: Normalized image after percentile-based clipping.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Calculate the lower and upper percentile values
    lower_val = np.percentile(image, lower_percentile)
    upper_val = np.percentile(image, upper_percentile)

    # Clip the image between the lower and upper percentile values
    clipped_image = np.clip(image, lower_val, upper_val)

    # Normalize to range [0, 1]
    normalized_image = (clipped_image - lower_val) / (upper_val - lower_val)

    return normalized_image


def plot_g_prob_distribution_w_skewness(
    g_probs, suffix, skewness_value, save_plot=True
):
    """
    Plot the distribution of probabilities with skewness annotation.

    Args:
        g_probs (array): Array of probabilities.
        suffix (str): Suffix for the plot labels.
        skewness_value (float): Skewness value to display.
        save_plot (bool): Whether to save the plot.
    """
    counts, _, _ = plt.hist(
        g_probs,
        bins=np.linspace(0, 1, 31),
        color="orange",
        alpha=0.75,
        edgecolor="grey",
    )

    plt.grid(True)

    max_height = max(counts)

    # Add text annotation for skewness value
    plt.text(
        0.5,
        max_height * 0.95,
        f"Skewness: {skewness_value:.2f}",
        fontsize=12,
        color="black",
        ha="center",
    )

    plt.xlabel(r"$P(y=1 \mid \mathbf{x}_" + "{" + suffix + "}" + ")$")
    plt.ylabel("Frequency")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_plot:
        plt.savefig(
            f"./plots/g_prob_distribution_{suffix}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
    plt.show()


def plot_histograms_for_countries(country_results, metric="accuracy", num_cols=5):
    """
    Plot histograms of prediction probabilities for multiple countries.

    Args:
        country_results (dict): Results containing probabilities and metrics for countries.
        metric (str): Metric to annotate on the plots (default is "accuracy").
        num_cols (int): Number of columns in the grid layout (default is 5).
    """
    all_countries = list(country_results.keys())
    num_countries = len(all_countries)
    num_rows = (num_countries + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    # Loop over all countries and create a histogram for each
    for i, country in enumerate(all_countries):
        print("Country:", country)

        ax = axes[i]

        g_pred_probs_testsetid = country_results[country]["g_pred_probs_testsetid"]

        # Calculate counts below and above 0.5
        below_0_5 = np.sum(g_pred_probs_testsetid < 0.5)
        above_0_5 = np.sum(g_pred_probs_testsetid >= 0.5)

        # Calculate percentages
        below_0_5_percentage = (below_0_5 / len(g_pred_probs_testsetid)) * 100
        above_0_5_percentage = (above_0_5 / len(g_pred_probs_testsetid)) * 100

        counts, _, _ = ax.hist(
            g_pred_probs_testsetid,
            bins=np.linspace(0, 1, 31),
            alpha=0.75,
            edgecolor="grey",
        )

        # Add a vertical dashed line at 0.5
        ax.axvline(x=0.5, color="gray", linestyle="--")

        max_height = max(counts)

        ax.text(
            0.25,
            max_height * 0.95,
            f"{below_0_5} ({below_0_5_percentage:.2f}%)",
            fontsize=12,
            color="black",
            ha="center",
        )
        ax.text(
            0.75,
            max_height * 0.95,
            f"{above_0_5} ({above_0_5_percentage:.2f}%)",
            fontsize=12,
            color="black",
            ha="center",
        )

        # Set title with country name
        ax.set_title(
            r"g on testset --" + "$P(y=1 \mid \mathbf{x}_" + "{" + country + "}" + ")$"
        )

        # Get the specified metric for the current country
        if metric in country_results[country]["metrics"]:
            metric_value = country_results[country]["metrics"][metric]
        else:
            metric_value = None

        # Add metric as text on the subplot if it exists
        if metric_value is not None:
            ax.text(
                0.5,
                0.4,
                f"{metric.capitalize()}: {metric_value:.4f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

        ax.set_xlabel(r"$P(y=1 \mid \mathbf{x}_" + "{" + country + "}" + ")$")
        ax.set_ylabel("Frequency")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Histogram Plots for All Countries with {metric.capitalize()}", fontsize=16
    )

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.show()


def plot_ftwtest_f1_skew_r2(df, save_plot=False):
    """
    Plot scatterplots of skewness against performance metrics with R² line.

    Args:
        df (pd.DataFrame): DataFrame containing skewness and metrics.
        save_plot (bool): Whether to save the plot.
    """
    plt.rcParams.update({"font.size": 16})
    metrics_to_plot = ["f1", "accuracy", "precision", "recall"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 10))
        ax = sns.scatterplot(
            x="skew_prob",
            y=metric,
            data=df,
            label="Countries",
            color="darkgreen",
            s=100,  # Marker size
        )

        ax.set_ylim(0.3, 1)

        x_ticks = ax.get_xticks()
        x_ticks = sorted(set(x_ticks.tolist() + [0]))
        ax.set_xticks(x_ticks)

        ax.set_xlim(left=0)

        ax.grid(False)

        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")

        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)

        plt.xlabel(r"Skewness of $P(y = 1 \mid g)$", fontsize=18)
        plt.ylabel(r"$F_1$ score of $f$ on FTW test set", fontsize=18)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        x = df["skew_prob"]
        y = df[metric]
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = np.corrcoef(x, y)[0, 1] ** 2

        x_sorted = np.sort(x)
        plt.plot(
            x_sorted,
            slope * x_sorted + intercept,
            linestyle=(0, (5, 5)),
            color="grey",
            linewidth=1.5,
            label=rf"Best Fit (R$^2$ = {r_squared:.2f})",
        )

        plt.legend(fontsize=18, loc="lower right")

        offset_x = 0.3
        offset_y = 0.0
        texts = []
        for i in range(df.shape[0]):
            texts.append(
                ax.text(
                    df["skew_prob"][i] + offset_x,
                    df[metric][i] + offset_y,
                    df["country"][i],
                    fontsize=18,
                )
            )

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="grey"))

        if save_plot:
            plt.savefig(
                f"./plots/skewness_vs_{metric}.pdf", bbox_inches="tight", dpi=300
            )
        plt.show()


def plot_ID_WILD(
    ID_all,
    WILD_all,
    save=False,
    file_format="svg",
    dpi=200,
    filepath="./",
    filename="geospatial_plot",
):
    """
    Plot geospatial coordinates for ID and WILD samples on an Equal Earth projection.

    Args:
        ID_all (dict): Dictionary containing coordinates for ID samples.
        WILD_all (dict): Dictionary containing coordinates for WILD samples.
        save (bool): Whether to save the plot.
        file_format (str): File format for saving the plot.
        dpi (int): DPI for the saved plot.
        filepath (str): Directory to save the plot.
        filename (str): Name of the file to save.
    """
    fontsize = 14
    plt.rcParams.update({"font.size": fontsize})

    id_coords = np.array(
        [[coord[0].item(), coord[1].item()] for coord in ID_all["coords"]]
    )

    # Separate latitudes and longitudes
    id_lats = id_coords[:, 0]
    id_lons = id_coords[:, 1]

    # Ensure WILD_all['coords'] is a NumPy array
    wild_coords = np.array(
        [[coord[0].item(), coord[1].item()] for coord in WILD_all["coords"]]
    )

    wild_lats = wild_coords[:, 0]
    wild_lons = wild_coords[:, 1]

    sns.set(style="whitegrid", palette="pastel")

    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.EqualEarth()}
    )

    # Add coastlines at 110 m resolution, can be changed to 10 and 50 m
    ax.coastlines("110m", "grey", lw=1)
    ax.gridlines()
    ax.set_global()

    # Add Natural Earth relief raster
    ax.stock_img()

    # Plot ID coordinates
    ax.scatter(
        id_lons,
        id_lats,
        transform=ccrs.PlateCarree(),
        marker="^",
        color="mediumorchid",
        label="In-distribution",
        s=10,
        alpha=1,
        edgecolor=None,
        linewidth=0.5,
    )

    # Plot WILD coordinates
    ax.scatter(
        wild_lons,
        wild_lats,
        transform=ccrs.PlateCarree(),
        marker="o",
        color="green",
        label="WILD",
        s=10,
        alpha=1,
        edgecolor=None,
        linewidth=0.5,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="WILD",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="mediumorchid",
            markersize=10,
            label="In-distribution",
        ),
    ]

    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(-0.1, 0.5),
        frameon=True,
        fancybox=True,
        fontsize=10,
    )

    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    if save:
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(
            os.path.join(filepath, f"{filename}.{file_format}"),
            format=file_format,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=dpi,
        )

    plt.tight_layout()
    plt.show()


def plot_ID_surrID_surrOOD(
    WILD_all,
    ID_all,
    save=False,
    file_format="pdf",
    dpi=200,
    filepath="./",
    filename="geospatial_plot",
):
    """
    Plot geospatial predictions for ID and WILD samples with predictions on an Equal Earth map.

    Args:
        WILD_all (dict): WILD coordinates and predictions.
        ID_all (dict): In-distribution coordinates.
        save (bool): Whether to save the plot.
        file_format (str): File format for saving the plot.
        dpi (int): DPI for the saved plot.
        filepath (str): Directory to save the plot.
        filename (str): Name of the file to save.
    """
    plt.rcParams.update({"font.size": 14})

    lats = []
    lons = []
    colors = []
    edge_colors = []
    markers = []

    for idx, coord in enumerate(WILD_all["coords"]):
        lat, lon = coord[0], coord[1]
        lats.append(float(lat))
        lons.append(float(lon))
        value = WILD_all["g_pred_probs"][idx]
        binary_dict_probs_to_binary = np.where(value > 0.5, 1, 0)
        colors.append("#FF9933" if binary_dict_probs_to_binary == 1 else "#1BA1E2")
        edge_colors.append("#FF9933" if binary_dict_probs_to_binary == 1 else "#1BA1E2")
        markers.append("o")

    sns.set(style="whitegrid", palette="pastel")

    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.EqualEarth()}
    )

    # Add coastlines and global features
    ax.coastlines("110m", "grey", lw=1)
    ax.gridlines()
    ax.set_global()
    ax.stock_img()

    # Plot coordinates with different colors and markers based on the key
    for lat, lon, color, marker, edge_color in zip(
        lats, lons, colors, markers, edge_colors
    ):
        ax.scatter(
            lon,
            lat,
            transform=ccrs.PlateCarree(),
            marker=marker,
            color=color,
            label="Coordinates",
            s=10,
            alpha=1,
            edgecolor=None,
            linewidth=1,
        )

    id_coords = np.array(ID_all["coords"])

    # Separate latitudes and longitudes directly
    id_lats = id_coords[:, 0]  # All latitude values
    id_lons = id_coords[:, 1]  # All longitude values

    ax.scatter(
        id_lons,
        id_lats,
        transform=ccrs.PlateCarree(),
        marker="^",
        color="mediumorchid",
        label="In-distribution",
        s=10,
        alpha=1,
        edgecolor=None,
        linewidth=0.5,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1BA1E2",
            markersize=10,
            label=r"$\mathit{g} \text{ Prediction: In-distribution}$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF9933",
            markersize=10,
            label=r"$\mathit{g} \text{ Prediction: Out-of-distribution}$",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="mediumorchid",
            markersize=10,
            label="In-distribution",
        ),
    ]

    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(-0.1, 0.5),
        frameon=True,
        fancybox=True,
        fontsize=10,
    )

    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    if save:
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(
            os.path.join(filepath, f"{filename}.{file_format}"),
            format=file_format,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=dpi,
        )

    plt.tight_layout()
    plt.show()


def plot_tsne(X, y, y_clustered):
    """
    Plot t-SNE visualizations for original and clustered labels.

    Args:
        X (array): Input feature matrix for t-SNE.
        y (array): Original labels.
        y_clustered (array): Clustered labels.
    """
    if y is not None and y_clustered is not None:
        cmap_discrete = plt.cm.get_cmap("coolwarm", 2)  # 2 colors for 0, and 2
        colors_discrete = [
            cmap_discrete(0),
            cmap_discrete(1),
        ]  # Three discrete colors for 0, 2->1, and 2->0

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Plot t-SNE with original labels (y)
        scatter1 = axes[0].scatter(
            X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap_discrete, alpha=1
        )
        axes[0].set_title("t-SNE with Original Labels (y)")
        axes[0].set_xlabel("t-SNE Component 1")
        axes[0].set_ylabel("t-SNE Component 2")
        fig.colorbar(scatter1, ax=axes[0], label="Original Label", ticks=np.arange(2))
        scatter1.set_clim(
            -0.5, 2.5
        )  # Set the color limits to match the discrete colormap
        scatter1.set_facecolor(
            colors_discrete
        )  # Set the facecolor to match the discrete colormap

        cmap_clustered = plt.cm.get_cmap(
            "coolwarm", 3
        )  # Three colors for 0, 2->1, and 2->0
        colors_clustered = [
            cmap_clustered(0),
            "yellow",
            "green",
        ]  # Three colors for 0, 2->1, and 2->0

        # Plot t-SNE with clustered labels (y_clustered)
        scatter2 = axes[1].scatter(
            X_tsne[:, 0], X_tsne[:, 1], c=y_clustered, cmap=cmap_clustered, alpha=1
        )
        axes[1].set_title("t-SNE with Clustered Labels (y_clustered)")
        axes[1].set_xlabel("t-SNE Component 1")
        axes[1].set_ylabel("t-SNE Component 2")
        fig.colorbar(scatter2, ax=axes[1], label="Clustered Label", ticks=np.arange(3))
        scatter2.set_clim(-0.5, 2.5)
        scatter2.set_facecolor(colors_clustered)

        plt.tight_layout()
        plt.show()
    else:
        print("Variable 'y' or 'y_clustered' is not defined.")


def plot_wild_coordinates(
    wild_coords_all,
    save=False,
    file_format="pdf",
    dpi=200,
    filepath="./",
    filename="g_wild_map_equal_earth",
):
    """
    Plot WILD coordinates on an Equal Earth projection.

    Args:
        wild_coords_all (list): List of WILD coordinates (latitude, longitude).
        save (bool): Whether to save the plot.
        file_format (str): File format for saving the plot.
        dpi (int): DPI for the saved plot.
        filepath (str): Directory to save the plot.
        filename (str): Name of the file to save.
    """
    fontsize = 14
    plt.rcParams.update({"font.size": fontsize})

    wild_lats = [coord[0].item() for coord in wild_coords_all]
    wild_lons = [coord[1].item() for coord in wild_coords_all]

    sns.set(style="whitegrid", palette="pastel")

    # Figure and axis with the Equal Earth projection
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.EqualEarth()}
    )

    # Coastlines at 110 m resolution, can be changed to 10 and 50 m
    ax.coastlines("110m", "grey", lw=1)
    ax.gridlines()
    ax.set_global()

    ax.stock_img()

    # Plot WILD coordinates
    ax.scatter(
        wild_lons,
        wild_lats,
        transform=ccrs.PlateCarree(),
        marker="^",
        color="mediumpurple",
        label="WILD",
        s=50,
        alpha=1,
        edgecolor="white",
        linewidth=0.5,
    )

    plt.legend(
        loc="lower left", fancybox=True, fontsize=12, markerscale=1.5, borderaxespad=1.5
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save:
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(
            os.path.join(filepath, f"{filename}.{file_format}"),
            format=file_format,
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
            transparent=True,
        )
    plt.show()
