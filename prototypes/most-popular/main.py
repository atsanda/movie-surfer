import pickle
from pathlib import Path
from sys import version_info

import click
import cloudpickle
import mlflow
import mlflow.pyfunc
import pandas as pd

from moviesurfer.metrics.classification import MeanAveragePrecisionTopK, PrecisionTopK
from moviesurfer.models import MostPopular

PROTOTYPE_NAME = "most-popular"

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        "pip",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                "moviesurfer @ https://github.com/atsanda/movie-surfer",
                f"cloudpickle=={cloudpickle.__version__}",
            ],
        },
    ],
    "name": "moviesurfer_env",
}


class MlFlowModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: dict[str, int | list[int]]) -> list[int]:
        return self.model.predict(**model_input)


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "data_folder", envvar="MOVIE_SURFER_DATA_FOLDER", type=click.Path(exists=True)
)
@click.argument(
    "models_folder", envvar="MOVIE_SURFER_MODELS_FOLDER", type=click.Path(exists=True)
)
def train(data_folder, models_folder):
    mlflow.set_experiment(PROTOTYPE_NAME)
    with mlflow.start_run():
        train_data = pd.read_csv(
            Path(data_folder) / "processed" / "ml-25m" / "ratings_train.csv"
        )

        model = MostPopular(top_k=100, movie_column="movieId", user_column="userId")
        mlflow.log_params(model.get_params())
        model.fit(train_data)

        model_save_folder = Path(models_folder) / PROTOTYPE_NAME
        if not model_save_folder.exists():
            model_save_folder.mkdir(parents=True, exist_ok=True)
        file_path = model.save(path=model_save_folder)

        model_info = mlflow.pyfunc.log_model(
            artifact_path=PROTOTYPE_NAME,
            python_model=MlFlowModelWrapper(),
            artifacts={"model": file_path},
            conda_env=conda_env,
            registered_model_name=PROTOTYPE_NAME,
        )

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

        pred = (
            train_data.groupby("userId")["movieId"]
            .apply(list)
            .apply(lambda item: loaded_model.predict(dict(n=5, seen_movies=item)))
        )
        pred = pd.DataFrame(pred)
        pred = pred.explode("movieId")
        pred["rank"] = pred.groupby("userId").cumcount() + 1
        pred = pred.reset_index()

        test_data = pd.read_csv(
            Path(data_folder) / "processed" / "ml-25m" / "ratings_test.csv"
        )

        mean_average_precision = MeanAveragePrecisionTopK(
            k=5, user_column="userId", movie_column="movieId", rank_column="rank"
        )
        map_value = mean_average_precision(pred, test_data)
        mlflow.log_metric("MAP-5", map_value)

        precision = PrecisionTopK(
            k=5, user_column="userId", movie_column="movieId", rank_column="rank"
        )
        precision_value = precision(pred, test_data)
        mlflow.log_metric("Precision-5", precision_value)


if __name__ == "__main__":
    cli()
