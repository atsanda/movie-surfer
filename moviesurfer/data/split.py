import logging
import os
from datetime import timedelta
from typing import Optional

import pandas as pd

from .utils import filename_from_path

logger = logging.getLogger(__name__)


def train_test_split_in_time(
    user_movies_interactions_csv: str,
    output_folder: str,
    offset_value: int,
    timestamp_column: Optional[str] = "timestamp",
    offset_units: Optional[str] = "days",
):
    """Splits a dataset into training and testing sets based on a specified time offset.
    New files are saved with '_train' and '_test' suffixes.

    Args:
        user_movies_interactions_csv (str): The path to the input CSV file
            containing user-movie interactions.
        output_folder (str): The path to the output folder where the
            split data will be saved.
        offset_value (int): The numerical value of the time offset.
        timestamp_column (str, optional): The name of the timestamp column
            in the CSV file. Defaults to "timestamp".
        offset_units (str, optional): The units of the time offset
            (e.g., "days", "hours", "minutes"). Defaults to "days".

    Example:
        >>> train_test_split_in_time(
                user_movies_interactions_csv="/path/to/interactions.csv",
                output_folder="/path/to/output",
                offset_value=7,
                timestamp_column="timestamp",
                offset_units="days"
            )
    """
    df = pd.read_csv(user_movies_interactions_csv)
    logger.debug(f"Data is loaded from {user_movies_interactions_csv}.")
    timestamps = pd.to_datetime(df[timestamp_column], unit="s")
    logger.info(
        f"Min timestamp: {timestamps.min()}, max timestamp: {timestamps.max()}."
    )

    dt = timedelta(**{offset_units: offset_value})

    split_timestamp = timestamps.max().normalize() - dt
    logger.info(f"Split timestamp: {split_timestamp}.")

    train = df.loc[timestamps < split_timestamp]
    test = df.loc[~df.index.isin(train.index)]
    logger.info(f"#rows in train: {len(train)}, #rows in test: {len(test)}")

    original_file_name = filename_from_path(user_movies_interactions_csv)

    for data, suffix in zip([train, test], ["train", "test"]):
        output_file_path = os.path.join(
            output_folder, f"{original_file_name}_{suffix}.csv"
        )
        data.to_csv(output_file_path, index=False)
        logger.info(f"{suffix} data is saved to {output_file_path}")
