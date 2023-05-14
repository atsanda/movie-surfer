import click

from .download import download
from .split import train_test_split


@click.group()
def data():
    pass


data.add_command(download)
data.add_command(train_test_split)
