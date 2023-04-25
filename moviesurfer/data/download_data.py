import logging
import os
import zipfile

import wget

logging.basicConfig(format="", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        raise Exception("Data already exists in the output directory.")

    try:
        logger.info("Downloading the MovieLens 25M Dataset...")
        data_zip = wget.download(url, out=output_dir)

        with zipfile.ZipFile(data_zip, "r") as data_ref:
            data_ref.extractall(output_dir)

        os.remove(data_zip)

        logger.info("Data downloaded successfully.")

    # Adressing temp files that are created if WGET is interrupted
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupting...")
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
        for file_name in os.listdir(parent_dir):
            if file_name.endswith(".tmp"):
                file_path = os.path.join(parent_dir, file_name)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
