# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing utilities for processing ID and WILD sets."""

import io
import os
import random
import time
import warnings
from datetime import datetime

import azure.storage.blob
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import stackstac
from shapely.geometry import Polygon
from tqdm import tqdm
from utils import load_config


def is_land_tile(item):
    """
    Determines if the given Sentinel-2 item is mostly land or sea.

    Parameters:
    - item: A STAC item representing a Sentinel-2 tile.

    Returns:
    - True if the tile is mostly land, False if it is mostly sea.
    """
    # Download and analyze the SCL band to determine if the tile is on land or sea
    scl_asset = item.assets["SCL"]
    scl_href = planetary_computer.sign(scl_asset.href)

    # Read the SCL band
    with rasterio.open(scl_href) as src:
        scl_data = src.read(1)  # Read the first band

    # Define SCL classes
    water_class = 7
    land_classes = [5, 6]  # Vegetation and not vegetated

    # Analyze the SCL band
    water_pixels = np.sum(scl_data == water_class)
    land_pixels = np.sum(np.isin(scl_data, land_classes))

    return land_pixels > water_pixels


def get_hemisphere(lat):
    """Determine the hemisphere based on latitude."""
    return "Northern" if lat >= 0 else "Southern"


def get_date_ranges(hemisphere):
    """Returns planting and harvesting date ranges depending on the hemisphere."""
    if hemisphere == "Northern":
        planting_start, planting_end = "04-01", "06-30"
        harvesting_start, harvesting_end = "09-01", "11-30"
    else:
        planting_start, planting_end = "10-01", "12-31"
        harvesting_start, harvesting_end = "03-01", "05-31"
    return (planting_start, planting_end), (harvesting_start, harvesting_end)


def extract_tile_from_metadata(item):
    """Extracts the MGRS tile identifier from the product metadata."""
    return item.properties["s2:mgrs_tile"]  # Use metadata to get the tile ID


def search_products_for_tile(catalog, tile_id, date_range, limit=10):
    """Search for Sentinel-2 product IDs for a specific MGRS tile and date range."""
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        datetime=date_range,
        query={"s2:mgrs_tile": {"eq": tile_id}},
        limit=limit,
    )

    # Fetch the items and convert them to a list
    items = search.item_collection()
    return [item.id for item in items] if items else []


def is_valid_pair(prod_id1, prod_id2):
    """
    Check if two product IDs form a valid pair based on the MGRS code.
    A valid pair should have:
    - The same MGRS code, which is the second to last segment of the product ID
    """
    mgrs_code1 = prod_id1.split("_")[-2]
    mgrs_code2 = prod_id2.split("_")[-2]

    return mgrs_code1 == mgrs_code2


def get_product_ids_for_planting_and_harvesting(prod_id1, catalog, max_retries=5):
    product_ids = []
    num_retries = 0
    item = None

    collection = catalog.get_collection("sentinel-2-l2a")
    # Retry logic for retrieving the item
    while num_retries < max_retries:
        try:
            item = collection.get_item(prod_id1)
            num_retries += 1
            if item:
                break
        except Exception as e:
            print(f"Error occurred: {e}")
            num_retries += 1
            if num_retries < max_retries:
                print(f"Retrying {prod_id1}, attempt {num_retries}")
                time.sleep(2**num_retries)  # Exponential backoff
            else:
                print(
                    f"Failed to get item after {max_retries} retries for product ID: {prod_id1}"
                )
                return (
                    product_ids  # Return an empty list indicating failure to get item
                )

    if item is None:
        print(
            f"Unable to retrieve item for product ID: {prod_id1} after {max_retries} attempts."
        )
        return product_ids

    try:
        bbox = item.geometry["coordinates"][0]
        polygon = Polygon(bbox)

        # Get the centroid of the polygon to determine hemisphere
        lat_centroid = polygon.centroid.y
        hemisphere = get_hemisphere(lat_centroid)

        # Get planting and harvesting date ranges
        planting_date_range, harvesting_date_range = get_date_ranges(hemisphere)
        current_year = datetime.now().year

        planting_date_range = f"{current_year}-{planting_date_range[0]}/{current_year}-{planting_date_range[1]}"
        harvesting_date_range = f"{current_year}-{harvesting_date_range[0]}/{current_year}-{harvesting_date_range[1]}"

        # Extract tile using metadata
        tile_id = extract_tile_from_metadata(item)

        # Search for products for the tile and date ranges
        planting_products = search_products_for_tile(
            catalog, tile_id, planting_date_range
        )
        if planting_products:
            prod_id_planting = planting_products[0]  # Select the first product
        else:
            raise ValueError(
                f"No planting products found for tile {tile_id} in {planting_date_range}."
            )

        harvesting_products = search_products_for_tile(
            catalog, tile_id, harvesting_date_range
        )
        if harvesting_products:
            prod_id_harvesting = harvesting_products[0]  # Select the first product
        else:
            raise ValueError(
                f"No harvesting products found for tile {tile_id} in {harvesting_date_range}."
            )

        return prod_id_planting, prod_id_harvesting

    except Exception as e:
        print(f"An error occurred while processing the item: {e}")

    return product_ids


def get_product_id_pairs(df, catalog, num_pairs, max_retries=5):
    """
    Extracts product IDs from a DataFrame, and collects pairs of planting and harvesting IDs.

    Args:
        df (pd.DataFrame): DataFrame containing product information with 'id' column.
        num_pairs (int): Number of pairs to collect.
        max_retries (int): Maximum number of retries for each product ID retrieval.

    Returns:
        List of tuples: Each tuple contains a pair of product IDs (planting, harvesting).
    """
    product_pairs = []
    num_error_hits = 0

    # Filter the DataFrame to ensure valid product IDs
    valid_rows = df[(df["land_mask"] == True) & (df["eo:cloud_cover"] < 10)]

    num_rows = valid_rows.shape[0]

    if num_rows < 2:
        raise ValueError("Not enough rows in the DataFrame to form pairs.")

    # Sample indices randomly without replacement
    sampled_indices = valid_rows.sample(n=num_rows, replace=False).index.tolist()
    progress_bar = tqdm(total=num_pairs, desc="Processing pairs", unit="pair")

    idx = 0
    while len(product_pairs) < num_pairs and idx < num_rows:
        try:
            # Extract the product ID (prod_id) from the DataFrame
            prod_id = valid_rows.loc[sampled_indices[idx]]["id"]
            # Get the pair of planting and harvesting IDs using the provided function
            product_ids = get_product_ids_for_planting_and_harvesting(
                prod_id, catalog, max_retries=max_retries
            )
            if len(product_ids) == 2:
                product_pairs.append((product_ids[0], product_ids[1]))
                progress_bar.update(1)

        except Exception as e:
            print(f"An error occurred while processing the product ID {prod_id}: {e}")
            num_error_hits += 1
            # Optionally, you could limit how many errors can happen before stopping the loop
            if num_error_hits > max_retries:
                print(
                    f"Exceeded maximum number of error retries ({max_retries}). Stopping."
                )
                break
            continue

        idx += 1

    progress_bar.close()

    if len(product_pairs) < num_pairs:
        print(
            f"Warning: Only {len(product_pairs)} pairs were found out of the requested {num_pairs}."
        )

    return product_pairs


def retrieve_metadata_and_save_parquet(product_ids, collection, parquet_output_file):
    """
    Retrieve metadata for a list of product IDs and save it into a Parquet file.

    Parameters:
    - product_ids: list of product IDs to retrieve metadata for
    - collection: the STAC collection object
    - parquet_output_file: file path to save the Parquet file
    """
    # Initialize an empty list to store the metadata
    metadata_list = []

    # Iterate over each product ID to retrieve the metadata
    for prod_id in tqdm(product_ids):
        try:
            # Make sure prod_id is not empty and is treated as a whole string
            prod_id = prod_id.strip()
            if not prod_id:
                print("Empty product ID encountered, skipping...")
                continue

            # Retrieve the item using the collection.get_item function
            item = collection.get_item(prod_id)
            if item is None:
                print(f"Item not found for product ID: {prod_id}")
                continue

            # Extract relevant metadata
            geometry = item.geometry  # or extract the actual geometry as needed
            datetime = item.datetime
            eo_cloud_cover = item.properties["eo:cloud_cover"]

            # Append to metadata list
            metadata_list.append(
                {
                    "id": prod_id,
                    "geometry": geometry,
                    "datetime": datetime,
                    "eo:cloud_cover": eo_cloud_cover,
                }
            )

        except Exception as e:
            print(f"Error retrieving data for product ID {prod_id}: {e}")

    # Convert the metadata list to a pandas DataFrame
    df = pd.DataFrame(metadata_list)

    initial_count = len(df)

    # Step 1: Filter out rows with eo:cloud_cover > 10
    filtered_df = df[df["eo:cloud_cover"] <= 10].copy()

    # Step 2: Identify MGRS codes of rows that were removed
    removed_mgrs_codes = (
        df[df["eo:cloud_cover"] > 10]["id"].apply(lambda x: x.split("_")[4]).unique()
    )

    # Step 3: Remove rows with the identified MGRS codes
    final_df = filtered_df[
        ~filtered_df["id"].apply(lambda x: x.split("_")[4]).isin(removed_mgrs_codes)
    ]

    # Calculate the number of affected and deleted products
    affected_count = len(df) - len(filtered_df)
    deleted_count = len(filtered_df) - len(final_df)
    total_deleted_count = initial_count - len(final_df)

    # Save the DataFrame to a Parquet file
    final_df.to_parquet(parquet_output_file, engine="pyarrow", compression="snappy")
    print(f"Metadata saved to {parquet_output_file}")
    print(f"Initial number of products: {initial_count}")
    print(f"Number of products affected by cloud cover criteria: {affected_count}")
    print(f"Number of pairs removed based on MGRS code: {deleted_count}")
    print(f"Total number of products deleted: {total_deleted_count}")


def extract_patches_from_pair(
    df_filtered, collection, container_client, azure_filename
):
    """
    Function to extract patches from planting and harvesting product IDs with matching x, y coordinates.
    """
    num_retries = 0
    num_error_hits = 0
    num_empty_hits = 0

    unique_mgrs_codes = df_filtered["id"].apply(lambda x: x.split("_")[4]).unique()
    all_results = []
    processed_mgrs_codes = set()
    skipped_mgrs_codes = set()

    # Loop through unique MGRS codes and process pairs
    for mgrs_code in tqdm(unique_mgrs_codes):
        if mgrs_code in processed_mgrs_codes or mgrs_code in skipped_mgrs_codes:
            continue

        # Get the planting and harvesting products for this MGRS code
        planting_harvesting_df = df_filtered[df_filtered["id"].str.contains(mgrs_code)]

        if planting_harvesting_df.shape[0] % 2 == 1:
            print(f"Skipping MGRS code {mgrs_code} because its length is odd.")
            skipped_mgrs_codes.add(mgrs_code)
            continue

        prod_id1 = planting_harvesting_df.iloc[0]["id"]
        prod_id2 = planting_harvesting_df.iloc[1]["id"]

        # Ensure the pair is valid
        if not is_valid_pair(
            prod_id1, prod_id2
        ):  # Assuming is_valid_pair is defined elsewhere
            print(f"Skipping invalid pair for MGRS code {mgrs_code}")
            skipped_mgrs_codes.add(mgrs_code)
            continue

        random_number = random.randint(
            10000, 99999
        )  # Generate a random number for unique naming

        # Process the pair
        product_ids = [prod_id1, prod_id2]
        results = []
        skip_pair = False  # Flag to skip both products if one fails

        for idx, prod_id in enumerate(product_ids):
            if skip_pair:
                break

            unique_name = (
                f"{random_number}_{mgrs_code}_planting"
                if idx == 0
                else f"{random_number}_{mgrs_code}_harvesting"
            )
            # Attempt to get this item with progressive exponential backoff
            item = None
            for j in range(2):
                try:
                    item = collection.get_item(prod_id)
                    break
                except Exception as e:
                    print(e)
                    print(f"Retrying for {prod_id}", j)
                    num_retries += 1
                    time.sleep(2**j)
            if item is None:
                print(f"Failed to get item {prod_id}")
                num_error_hits += 1
                skip_pair = True
                continue

            # Stack the item
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stack = stackstac.stack(
                    item,
                    assets=["B04", "B03", "B02", "B08"],
                )
            _, num_channels, height, width = stack.shape
            # Ensure same x, y for both images in the pair
            if idx == 0:
                x = np.random.randint(0, width - 256)
                y = np.random.randint(0, height - 256)
            try:
                patch = stack[0, :, y : y + 256, x : x + 256].compute()
            except RuntimeError:
                print(f"Failed to read item {prod_id}")
                num_error_hits += 1
                skip_pair = True
                continue

            percent_empty = np.mean((np.isnan(patch.data)).sum(axis=0) == num_channels)
            percent_zero = np.mean((patch.data == 0).sum(axis=0) == num_channels)
            # Skip if patch is mostly empty or zero
            if percent_empty > 0.1 or percent_zero > 0.1:
                num_empty_hits += 1
                skip_pair = True
                continue

            # Save the patch
            with io.BytesIO() as buffer:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    patch = patch.astype(np.uint16)
                patch.rio.to_raster(
                    buffer, driver="COG", dtype=np.uint16, compress="LZW", predictor=2
                )
                buffer.seek(0)
                blob_client = container_client.get_blob_client(
                    f"{azure_filename}/{unique_name}.tif"
                )
                blob_client.upload_blob(buffer, overwrite=True)
            results.append((idx, prod_id, x, y))

        if skip_pair:
            skipped_mgrs_codes.add(mgrs_code)
            continue  # Skip saving patches and move to the next MGRS code

        # Collect the results only if both products are processed successfully
        all_results.extend(results)
        processed_mgrs_codes.add(mgrs_code)

    print("Summary:")
    print(f"num error hits: {num_error_hits}")
    print(f"num empty hits: {num_empty_hits}")
    print(f"num retries: {num_retries}")

    return all_results


def main():
    # Load configuration
    config = load_config()

    # Extract settings from configuration
    storage_account = config["storage_account"]
    container_name = config["container_name"]
    sas_key = config["sas_key"]
    num_pairs = config["num_pairs"]
    s2_parquet_fn = config["s2_parquet_fn"]
    new_parquet_fname = config["new_parquet_fname"]
    azure_filename = config["azure_filename"]

    # Initialize Azure container client
    container_client = azure.storage.blob.ContainerClient(
        storage_account, container_name=container_name, credential=sas_key
    )

    # Initialize STAC catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("sentinel-2-l2a")

    # Check if parquet file exists, if not process and create it
    if not os.path.exists(new_parquet_fname):
        print("Parquet file does not exist. Processing...")
        df = pd.read_parquet(s2_parquet_fn, engine="pyarrow")
        collected_product_pairs = get_product_id_pairs(
            df, catalog, num_pairs=num_pairs, max_retries=1
        )
        collected_product_pairs = [
            item for sublist in collected_product_pairs for item in sublist
        ]
        retrieve_metadata_and_save_parquet(
            collected_product_pairs, collection, new_parquet_fname
        )
    else:
        print(f"Parquet file {new_parquet_fname} already exists. Skipping processing.")

    # Load the parquet file and extract patches
    plant_harvest_df = pd.read_parquet(new_parquet_fname, engine="pyarrow")
    print("plant_harvest_df.head()", plant_harvest_df.head())
    results = extract_patches_from_pair(
        plant_harvest_df, collection, container_client, azure_filename
    )
    print("Patch extraction completed. Results:", results)


if __name__ == "__main__":
    main()
