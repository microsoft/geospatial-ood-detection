# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing utilities for processing ID and WILD sets."""

import os
import pickle

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from adjustText import adjust_text
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import skew
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_data_dict_as_pkl(wild_data_dict, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(wild_data_dict, f)
    print(f"Wild data dictionary saved to {file_path}")


def load_data_dict_from_pkl(file_path):
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def calculate_metrics(true_masks, predicted_masks):
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


def correlation_analysis(df):
    # Select only numeric columns for correlation calculation
    numeric_columns = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def plot_f_and_g_preds_probab(
    window_a, window_b, ood_label, output, filepath=None, filename=None
):
    # Create a GridSpec with custom width ratios and reduced horizontal spacing
    width_ratios = [4, 4, 0.5, 4]  # Aux column is 0.5 relative to image columns
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(
        1, 4, width_ratios=width_ratios, wspace=0.05
    )  # Set wspace to 0.05

    # Create subplots based on GridSpec
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    axes[0].imshow(window_a)
    axes[0].axis("off")

    axes[1].imshow(window_b)
    axes[1].axis("off")

    ax_aux = axes[2]
    ax_aux.axis("off")
    ax_aux.set_aspect("auto")

    # Define thermometer properties
    thermometer_width = 0.2
    thermometer_height = 0.6  # Reduced height to allow space for subtitle
    thermometer_x = 0.4  # Centered in the aux axis
    thermometer_y = 0.3  # Start from a slightly raised bottom to fit subtitle

    # Normalize the colormap to the range [0, 1]
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.bwr  # Blue-Red colormap

    # Create an array of values from 0 to 1 for the gradient
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)

    # Mask the gradient to fill only up to the ood_label value
    mask = np.ones_like(gradient)
    fill_index = int(ood_label * 256)
    mask[fill_index:] = np.nan  # Set the upper part to NaN to make it transparent

    # Plot the masked gradient as an image
    ax_aux.imshow(
        mask[::-1],
        aspect="auto",
        cmap=cmap,
        norm=norm,  # Reverse the gradient to fill from bottom to top
        extent=(
            thermometer_x,
            thermometer_x + thermometer_width,
            thermometer_y,
            thermometer_y + thermometer_height,
        ),
    )

    # Draw the thermometer outline
    outline = Rectangle(
        (thermometer_x, thermometer_y),
        thermometer_width,
        thermometer_height,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax_aux.add_patch(outline)

    # Add labels for 0 and 1
    ax_aux.text(
        thermometer_x - 0.05,
        thermometer_y,
        "0",
        ha="left",
        va="center",
        fontsize=12,
        color="black",
    )
    ax_aux.text(
        thermometer_x - 0.05,
        thermometer_y + thermometer_height,
        "1",
        ha="left",
        va="center",
        fontsize=12,
        color="black",
    )

    # Display the ood_label value next to the filled portion
    ax_aux.text(
        thermometer_x + thermometer_width + 0.05,
        thermometer_y + fill_index / 256 * thermometer_height,
        f"{ood_label:.2f}",
        ha="left",
        va="center",
        fontsize=12,
        color="black",
    )

    axes[3].imshow(output)
    axes[3].axis("off")

    # Finalize the layout
    plt.subplots_adjust(wspace=0.05)  # Ensure horizontal spacing is minimal

    # Save the plot if filepath and filename are provided, otherwise show the plot
    if filepath is not None and filename is not None:
        # Create the directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        # Save the plot without borders and with a transparent background
        plt.savefig(
            os.path.join(filepath, filename),
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(fig)  # Close the figure to free up memory
    else:
        # Show the plot
        plt.show()


def percentile_stretch(image, lower_percentile=2, upper_percentile=98):
    """
    Stretch the image values based on percentiles.

    Args:
    - image (numpy array or tensor): Input image to normalize.
    - lower_percentile (int): Lower bound percentile for clipping (default is 2).
    - upper_percentile (int): Upper bound percentile for clipping (default is 98).

    Returns:
    - normalized_image (numpy array or tensor): Image after percentile-based normalization.
    """
    # Convert the image to a numpy array if it's a tensor
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
    Plots the distribution of probabilities with a given skewness value as text annotation.

    Parameters:
    g_probs (numpy array): Array of probabilities.
    suffix (str): Suffix to be added.
    skewness_value (float): Skewness value to be displayed on the plot.
    """
    # Create the histogram with adjusted binning
    counts, bins, patches = plt.hist(
        g_probs,
        bins=np.linspace(0, 1, 31),
        color="orange",
        alpha=0.75,
        edgecolor="grey",
    )

    # Add grid to the plot
    plt.grid(True)

    # Determine the maximum y-value for text placement
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
            f"g_prob_distribution_{suffix}.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
    plt.show()


def plot_histograms_for_countries(country_results, metric="accuracy", num_cols=5):

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

        # Create the histogram with adjusted binning
        counts, bins, patches = ax.hist(
            g_pred_probs_testsetid,
            bins=np.linspace(0, 1, 31),
            alpha=0.75,
            edgecolor="grey",
        )

        # Add a vertical dashed line at 0.5
        ax.axvline(x=0.5, color="gray", linestyle="--")

        # Determine the maximum y-value for text placement
        max_height = max(counts)

        # Add text annotations for counts and percentages
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

        # Add labels
        ax.set_xlabel(r"$P(y=1 \mid \mathbf{x}_" + "{" + country + "}" + ")$")
        ax.set_ylabel("Frequency")

        # Remove the upper and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a main title to the figure
    fig.suptitle(
        f"Histogram Plots for All Countries with {metric.capitalize()}", fontsize=16
    )

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.show()


def visualize_data(df, save_plot=False):
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 8))
        ax = sns.scatterplot(
            x="skew_prob", y=metric, data=df, label="Countries", color="darkgreen"
        )

        # Add a horizontal line for the mean of the metric
        plt.axhline(y=df[metric].mean(), color="grey", linestyle="--", label="Mean")

        # Add corresponding negative x ticks
        x_ticks = ax.get_xticks()
        x_ticks = sorted(set(x_ticks.tolist() + [0]))
        ax.set_xticks(x_ticks)

        # Start x-axis from 0
        ax.set_xlim(left=0)

        # Add background grid
        ax.grid(True)

        plt.xlabel("Skewness of Probability Distribution")
        plt.ylabel(metric.capitalize())

        # Remove the top and right plot panes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.legend()

        # Annotate each point with the country name
        texts = []
        for i in range(df.shape[0]):
            texts.append(
                ax.text(df["skew_prob"][i], df[metric][i], df["country"][i], fontsize=9)
            )

        # Adjust text to minimize overlap
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="grey"))

        save_plot = True
        if save_plot:
            plt.savefig(f"skewness_vs_{metric}.svg", bbox_inches="tight", dpi=300)
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
    Plots geospatial data on an Equal Earth projection.

    Parameters:
    - ID_all: Dictionary containing 'coords' key with in-distribution coordinates as list of tuples [(lat, lon), ...].
    - WILD_all: Dictionary containing 'coords' key with WILD coordinates as list of tuples [(lat, lon), ...].
    - save: Boolean, whether to save the plot or not.
    - file_format: Format to save the plot ('pdf', 'png', etc.).
    - dpi: DPI for the saved plot.
    - filepath: Directory to save the plot.
    - filename: Name of the saved plot file without extension.
    """

    fontsize = 14
    plt.rcParams.update({"font.size": fontsize})

    id_lats = [coord[0].item() for coord in ID_all["coords"]]
    id_lons = [coord[1].item() for coord in ID_all["coords"]]
    wild_lats = [coord[0].item() for coord in WILD_all["coords"]]
    wild_lons = [coord[1].item() for coord in WILD_all["coords"]]

    # Set a seaborn style
    sns.set(style="whitegrid", palette="pastel")

    # Set up the figure and axis with the Equal Earth projection
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.EqualEarth()}
    )

    # Add coastlines at 110 m resolution, can be changed to 10 and 50 m
    ax.coastlines("110m", "grey", lw=1)
    ax.gridlines()
    ax.set_global()

    # Add Natural Earth relief raster
    ax.stock_img()

    # Plot ID coordinates (in blue)
    ax.scatter(
        id_lons,
        id_lats,
        transform=ccrs.PlateCarree(),
        marker="^",
        color="tomato",
        label="In-distribution",
        s=50,
        alpha=1,
        edgecolor="white",
        linewidth=0.5,
    )

    # Plot WILD coordinates (in red)
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

    # Add a legend with a fancy box
    plt.legend(
        loc="lower left", fancybox=True, fontsize=12, markerscale=1.5, borderaxespad=1.5
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout and remove whitespace
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
    Plots geospatial prediction data on an Equal Earth projection.

    Parameters:
    - WILD_all: Dictionary containing 'coords' key with WILD coordinates and 'g_pred_probs' with prediction probabilities.
    - ID_all: Dictionary containing 'coords' key with in-distribution coordinates.
    - save: Boolean, whether to save the plot or not.
    - file_format: Format to save the plot ('pdf', 'png', 'svg', etc.).
    - dpi: DPI for the saved plot.
    - filepath: Directory to save the plot.
    - filename: Name of the saved plot file without extension.
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
        colors.append(
            "#FF6347" if binary_dict_probs_to_binary == 1 else "#4682B4"
        )  # Use Tomato for OOD and SteelBlue for ID
        edge_colors.append(
            "#FF6347" if binary_dict_probs_to_binary == 1 else "#4682B4"
        )  # Use Tomato for OOD and SteelBlue for ID
        markers.append("o")

    # Set a seaborn style
    sns.set(style="whitegrid", palette="pastel")

    # Set up the figure and axis with the Equal Earth projection
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
            s=40,
            alpha=0.9,
            edgecolor=None,
            linewidth=1,
        )

    # Plot ID coordinates
    id_lats = [coord[0].item() for coord in ID_all["coords"]]
    id_lons = [coord[1].item() for coord in ID_all["coords"]]
    ax.scatter(
        id_lons,
        id_lats,
        transform=ccrs.PlateCarree(),
        marker="^",
        color="mediumorchid",
        label="In-distribution",
        s=40,
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
            markerfacecolor="#4682B4",
            markersize=10,
            label=r"$\text{g*} \text{ Prediction: In-distribution}$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF6347",
            markersize=10,
            label=r"$\text{g*} \text{ Prediction: Out-of-distribution}$",
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
    Plots t-SNE visualizations with original and clustered labels.

    Parameters:
    - X: The input data for t-SNE.
    - y: The original labels.
    - y_clustered: The clustered labels.
    """
    if y is not None and y_clustered is not None:
        # Define the colors for the discrete colormap
        cmap_discrete = plt.cm.get_cmap("coolwarm", 2)  # 2 colors for 0, and 2
        colors_discrete = [
            cmap_discrete(0),
            cmap_discrete(1),
        ]  # Blue for 0, Yellow for 2->1, Green for 2->0

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

        # Define the colors for the clustered colormap
        cmap_clustered = plt.cm.get_cmap(
            "coolwarm", 3
        )  # 3 colors for 0, 2->1, and 2->0
        colors_clustered = [
            cmap_clustered(0),
            "yellow",
            "green",
        ]  # Blue for 0, Yellow for 2->1, Green for 2->0

        # Plot t-SNE with clustered labels (y_clustered)
        scatter2 = axes[1].scatter(
            X_tsne[:, 0], X_tsne[:, 1], c=y_clustered, cmap=cmap_clustered, alpha=1
        )
        axes[1].set_title("t-SNE with Clustered Labels (y_clustered)")
        axes[1].set_xlabel("t-SNE Component 1")
        axes[1].set_ylabel("t-SNE Component 2")
        fig.colorbar(scatter2, ax=axes[1], label="Clustered Label", ticks=np.arange(3))
        scatter2.set_clim(
            -0.5, 2.5
        )  # Set the color limits to match the clustered colormap
        scatter2.set_facecolor(
            colors_clustered
        )  # Set the facecolor to match the clustered colormap

        # Show the plots
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
    Plots WILD coordinates on an Equal Earth projection.

    Parameters:
    - wild_coords_all: List of tuples [(lat, lon), ...] containing WILD coordinates.
    - save: Boolean, whether to save the plot or not.
    - file_format: Format to save the plot ('pdf', 'png', etc.).
    - dpi: DPI for the saved plot.
    - filepath: Directory to save the plot.
    - filename: Name of the saved plot file without extension.
    """

    fontsize = 14
    plt.rcParams.update({"font.size": fontsize})

    wild_lats = [coord[0].item() for coord in wild_coords_all]
    wild_lons = [coord[1].item() for coord in wild_coords_all]

    # Set a seaborn style
    sns.set(style="whitegrid", palette="pastel")

    # Set up the figure and axis with the Equal Earth projection
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.EqualEarth()}
    )

    # Add coastlines at 110 m resolution, can be changed to 10 and 50 m
    ax.coastlines("110m", "grey", lw=1)
    ax.gridlines()
    ax.set_global()

    # Add Natural Earth relief raster
    ax.stock_img()

    # Plot WILD coordinates (in red)
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

    # Add a legend with a fancy box
    plt.legend(
        loc="lower left", fancybox=True, fontsize=12, markerscale=1.5, borderaxespad=1.5
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout and remove whitespace
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
