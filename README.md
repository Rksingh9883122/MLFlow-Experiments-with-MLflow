# MLFlow-Experiments-with-MLflow
This repo has complete demo over performace tracking using ML flow
# 🍷 Wine Quality Prediction with MLflow & Random Forests

![Wine Classification](https://images.unsplash.com/photo-1595981267035-7b04ca84a82d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80)

## 📋 Experiment Overview
This project demonstrates an end-to-end machine learning workflow using MLflow for experiment tracking and model management. We predict wine quality categories using a Random Forest classifier trained on the classic Wine dataset.

**Key Features:**
- MLflow for experiment tracking and model versioning
- Automated logging of parameters, metrics and artifacts
- Visual evaluation with confusion matrix
- Reproducible machine learning pipeline

## 📂 Project Structure
```
mlflow-wine-experiment/
├── src/
│   └── wine_classifier.py      # Main training script
├── mlruns/                      # MLflow tracking data (auto-generated)
├── artifacts/                   # Model artifacts (auto-generated)
├── README.md                    # This documentation
└── requirements.txt             # Python dependencies
```

## ⚙️ Technical Setup

### Prerequisites
- Python 3.8+
- MLflow 2.0+
- Scikit-learn 1.0+
- Matplotlib & Seaborn

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/mlflow-wine-experiment.git
cd mlflow-wine-experiment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Start MLflow Tracking Server
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 \
  --port 5000
```

## 🚀 Running the Experiment
Execute the training script:
```bash
python src/wine_classifier.py
```

## 🔍 Viewing Results
Access the MLflow UI in your browser:
```
http://localhost:5000
```

## 📊 Experiment Details

### Dataset
The Wine dataset contains 178 samples with 13 chemical features, classified into 3 wine cultivars:
- Class 0: "class_0"
- Class 1: "class_1"
- Class 2: "class_2"

Features include alcohol content, malic acid, ash, alkalinity, magnesium, phenols, flavonoids, etc.

### Model Architecture
**Algorithm:** Random Forest Classifier  
**Hyperparameters:**
- `max_depth`: 10
- `n_estimators`: 10
- `random_state`: 42

### Evaluation Metrics
- **Accuracy**: 0.94
- **Confusion Matrix**:

| Actual \ Predicted | class_0 | class_1 | class_2 |
|-------------------|---------|---------|---------|
| **class_0**       | 6       | 0       | 0       |
| **class_1**       | 0        | 6       | 1       |
| **class_2**       | 0        | 0       | 5       |

## 📦 Model Artifacts
Each run logs:
- Trained model (MLflow format)
- Source code
- Confusion matrix visualization
- Hyperparameters and metrics
- Environment metadata

## 📈 Results Interpretation
The model achieves 94% accuracy on the test set. The confusion matrix shows:
- Perfect classification for class_0 wines
- 1 misclassification between class_1 and class_2
- Strong overall performance with limited data

## 🛠️ How to Use the Model
Load a logged model for predictions:
```python
import mlflow.pyfunc

model_uri = "runs:/<RUN_ID>/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Sample prediction
sample = [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]
print(loaded_model.predict([sample]))  # Output: [1]
```

## 📚 Learning Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [Wine Dataset Description](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)

## 🤝 Contribution
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎯 Key Takeaways:** This implementation showcases how MLflow brings structure and reproducibility to machine learning experiments while maintaining flexibility for data scientists to iterate quickly. The combination of automated tracking and visual evaluation creates a powerful workflow for model development.
