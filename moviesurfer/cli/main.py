import logging

import click

from moviesurfer.data.download_data import download_data

logger = logging.getLogger(__name__)


@click.group()
def cli():
    logger.info("moviesurfer cli started")


@cli.group()
def data():
    pass


@data.command()
def download():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    try:
        download_data()
    except FileExistsError:
        click.echo("No need to call 'download data', file exists")


if __name__ == "__main__":
    cli()
