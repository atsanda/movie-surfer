import click

from moviesurfer.data.download_data import download_data


@click.command()
def download():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    try:
        download_data(output_dir="data/raw")
    except FileExistsError:
        click.echo("No need to call 'download data', file exists")
