import mlflow
import os
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load environment variables from .env file
load_dotenv('.env')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set MLflow experiment
mlflow.set_experiment('mlflow_test15')

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Start MLflow run
with mlflow.start_run() as run:
    # Load movie recommendation dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Create and train a movie recommendation model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset
    predictions = knn.predict(X_test)

    # Log model artifacts
    mlflow.sklearn.log_model(knn, "movie_recommendation_model")

    # Log evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)

    # Display run information
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
