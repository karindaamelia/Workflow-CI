import argparse
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

# Argument parser to allow CLI configuration
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Configure MLflow tracking (remote if DagsHub credentials exist, otherwise local)
if os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME'):
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN')
    mlflow.set_tracking_uri("https://dagshub.com/karindaamelia/air-quality-model.mlflow")
    mlflow.set_experiment("air-quality-basic")
    print("Using remote MLflow tracking on DagsHub")
    use_remote = True
else:
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("air-quality-basic-local")
    print("DagsHub credentials not found. Using local MLflow tracking.")
    use_remote = False

# Load the dataset
data = pd.read_csv(args.data_path)

# Split features and target
X = data.drop(columns=["AH"])   # Features
y = data["AH"]                  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Train and evaluate the model within an MLflow run
with mlflow.start_run():
    model = RandomForestRegressor(random_state=args.random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)

    # Manually log
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("evs", evs)
    
    # Log run parameters
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features", X_train.shape[1])

    # Save the model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    mlflow.log_artifact(args.model_output)

    # Register the model in the MLflow Model Registry
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    try:
        mlflow.register_model(model_uri=model_uri, name="air-quality-prediction")
        print("Model registered successfully.")
    except Exception as e:
        print("Model registration skipped or failed:", e)

    # Display tracking information
    if use_remote:
        print("Tracking URL: https://dagshub.com/karindaamelia/air-quality-model.mlflow")
    else:
        print("Tracking directory: ./mlruns")