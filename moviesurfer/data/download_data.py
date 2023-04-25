import logging
import os
import zipfile

import wget


# Function to download data
def download_data(
    output_dir="moviesurfer/data",
    url="http://files.grouplens.org/datasets/movielens/ml-25m.zip",
):
    """
    Downloads the MovieLens 25M Dataset, extracts it, and deletes the zip file.

    :param output_dir: str, the output directory to save the downloaded data
    :return: None
    """
    logging.basicConfig(level=logging.DEBUG)

    # check if movies.csv file exists in output_dir
    movies_csv_file = os.path.join(output_dir, "ml-25m/movies.csv")

    if os.path.exists(movies_csv_file):
        logging.info("Data already exists in the output directory.")
        return

    logging.info("Downloading the MovieLens 25M Dataset...")
    filename = wget.download(url, out=output_dir)

    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(filename)

    logging.info("Data downloaded successfully.")
