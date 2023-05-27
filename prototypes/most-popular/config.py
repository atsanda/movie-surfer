import pickle
from sys import version_info

import cloudpickle
import mlflow
import mlflow.pyfunc

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
