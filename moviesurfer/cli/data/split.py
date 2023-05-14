import click

from moviesurfer.data.split import train_test_split_in_time


@click.command
def train_test_split():
    """
    Splits raw data into train and test
    """
    train_test_split_in_time(
        user_movies_interactions_csv="data/raw/ml-25m/ratings.csv",
        output_folder="data/processed/ml-25m",
        timestamp_column="timestamp",
        offset_units="days",
        offset_value=60,
    )
