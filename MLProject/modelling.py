import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset hasil preprocessing
data = pd.read_csv("namadataset_preprocessing/loan_data_cleaned.csv")

X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Loan_Status_Experiment")
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    mlflow.log_metric("manual_accuracy", acc)

print("Selesai dan tercatat di MLflow.")