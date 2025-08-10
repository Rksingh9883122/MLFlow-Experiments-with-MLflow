
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Rksingh9883122', repo_name='MLFlow-Experiments-with-MLflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Rksingh9883122/MLFlow-Experiments-with-MLflow.mlflow")



# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.10, random_state=42
)

# Hyperparameters
max_depth = 8
n_estimators = 5


# Experiment
mlflow.set_experiment("YT-MLOPS-Exp2")

with mlflow.start_run():
    # Train
    
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
        
    # Predict & evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
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
    
    # Log the model
    #mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    
    print(accuracy_score(y_test, y_pred))
    