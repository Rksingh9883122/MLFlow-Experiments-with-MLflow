import os
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"  # Override environment variable

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os

# Optional: Set tracking URI if using MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and hyperparameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Enable autologging
mlflow.sklearn.autolog()

# Start MLflow experiment
mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # Log each child run
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Log best model info
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    # Save and log training data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv("train.csv", index=False)
    mlflow.log_artifact("train.csv", artifact_path="training")

    # Save and log test data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv("test.csv", index=False)
    mlflow.log_artifact("test.csv", artifact_path="testing")

    # Log confusion matrix
    y_pred = grid_search.best_estimator_.predict(X_test)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log feature importance
    importances = grid_search.best_estimator_.feature_importances_
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    # Log source code (safe check for __file__)
    if '__file__' in globals():
        mlflow.log_artifact(__file__)

    # Log model
    #mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Set tags
    mlflow.set_tag("author", "Raj Singh")

    # Print results
    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_score)

