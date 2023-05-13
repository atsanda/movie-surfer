import logging
import os
import zipfile
from tempfile import TemporaryDirectory

import wget

logger = logging.getLogger(__name__)


# Function to download data
def download_data(
    output_dir="moviesurfer/data",
    url="http://files.grouplens.org/datasets/movielens/ml-25m.zip",
):
    """
    Downloads the MovieLens 25M Dataset, extracts it, and deletes the zip file.

    :param output_dir: str, the output directory to save the downloaded data
    :param url: str, the url address to download from
    :return: None
    """

    # Check if movies.csv file exists in output_dir
    movies_csv_file = os.path.join(output_dir, "ml-25m/movies.csv")

    if os.path.exists(movies_csv_file):
        raise FileExistsError("Data already exists in the output directory.")

    # Download Data while adressing temp files that are created if WGET is interrupted
    try:
        with TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_folder")
            logger.warning("Downloading the MovieLens 25M Dataset...")
            wget.download(url, temp_file)

            with zipfile.ZipFile(temp_file, "r") as data_ref:
                data_ref.extractall(output_dir)

            os.remove(temp_file)

        logger.warning("Data downloaded successfully.")

    except KeyboardInterrupt:
        logging.warning("\nThe download was interrupted.")
        if os.path.exists(temp_file):
            os.remove(temp_file)
