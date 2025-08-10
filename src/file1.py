import os
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"  # Override environment variable

import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Point to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Confirm port 5000

# Experiment
mlflow.set_experiment("YT-MLOPS-Exp2")

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.10, random_state=42
)

# Hyperparameters
params = {"max_depth": 10, "n_estimators": 10}

with mlflow.start_run() as run:
    # Train
    rf = RandomForestClassifier(random_state=42, **params)
    rf.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save & log the plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log script
    mlflow.log_artifact(__file__)
    
    # set Tag
    mlflow.set_tags({"Author":"Raj", "Project": "Wine Quality Prediction"})
    
    #log Model
    #mlflow.sklearn.log_model(rf, "model")

    print(accuracy_score(y_test, y_pred))
    