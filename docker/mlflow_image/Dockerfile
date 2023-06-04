FROM python:3.9

# Install required dependencies
RUN pip install mlflow scikit-learn psycopg2 boto3

# Create a directory for the MLflow
RUN mkdir /app

# Copy the MLflow script to the container
COPY mlflow_test.py /app

# Copy the .env file to the container
COPY .env /app

# Set the working directory
WORKDIR /app

# Expose port 5000 for the MLflow server
EXPOSE 5000

# Set environment variables from .env file
ENV $(cat .env | xargs)

# Execute the MLflow script
CMD ["python", "/mlflow_test.py"]
