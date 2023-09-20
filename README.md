# Breast Cancer Prediction Machine Learning Project

This project uses machine learning to predict breast cancer diagnosis based on the Wisconsin Breast Cancer dataset. It includes data preprocessing, exploratory data analysis, model training, and evaluation. The project is implemented in Python.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- `numpy`
- `matplotlib`
- `pandas`
- `seaborn`
- `scikit-learn`

You can install them using `pip`:

```bash
pip install numpy matplotlib pandas seaborn scikit-learn
```

## Getting Started

1. Clone the repository: 
```bash
git clone https://github.com/your-username/your-repo.git
```
2. Navigate to the project directory:
```bash
cd your-repo
```
3. Run the Python script:
```bash
python breast_cancer_prediction.py
```
## Project Structure
`breast_cancer_prediction.py`: The main Python script that performs data preprocessing, model training, and evaluation.
`data.csv`: The dataset file containing breast cancer data.
`BreastCancerPrediction.joblib`: The saved Random Forest model using joblib.

## Data Preprocessing

- The dataset is loaded from the `data.csv` file.
- Null columns are removed.
- Label encoding is applied to convert 'M' (Malignant) and 'B' (Benign) labels to 1 and 0, respectively.
- The dataset is split into independent (X) and dependent (Y) variables.
- Feature scaling is performed on the independent variables.

## Model/Algorithms

Three machine learning models are used in this project:
1. Logistic Regression
2. Decision Tree
3. Random Forest
Each model is trained on the preprocessed data.

## Model Evaluation
Model performance is evaluated using accuracy and classification reports. The evaluation is done on a test dataset.

## Results
The project provides predictions for breast cancer diagnosis using the trained Random Forest model. The predicted values are compared with the actual values.

## Save the Model
The trained Random Forest model is saved as `BreastCancerPrediction.joblib` for future use.
Feel free to explore the code and dataset to gain insights into breast cancer prediction.
Happy coding!






