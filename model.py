import argparse
import os
import mlflow
import pandas as pd
pd.set_option('display.max_columns',100)
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


import mlflow.sklearn

def train_model(data_path: str) -> None:
    df=pd.read_csv(data_path)

    upper_bound=df.quantile(q=.97,numeric_only=True)

    df = df[(df['Horizontal_Distance_To_Fire_Points'] <= upper_bound['Horizontal_Distance_To_Fire_Points'])]

    abs(df.corr(numeric_only=True)['Cover_Type'].sort_values(ascending=False))

    x=df.drop(['Cover_Type','Id'],axis=1)
    y=df[['Cover_Type']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        local_uri = "http://localhost:5000"
        print(f"MLflow tracking URI not set. Using local store at {local_uri}")
        mlflow.set_tracking_uri(local_uri)
        
    with mlflow.start_run():
        rf = RandomForestClassifier()
        model = rf.fit(x_train, y_train)

        preds = model.predict(x_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Accuracy: {accuracy}")

        f1 = f1_score(y_test, preds, average='weighted')
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        print(f"F1 score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Для подробного отчета по классам:
        print(classification_report(y_test, preds))

        mlflow.log_param("n_estimators", model.n_estimators)

        mlflow.log_metrics({"accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1})

        mlflow.set_tags({
            "model": "RandomForest",
            "version": "1.0",
            "type": "training"
        })
        
        mlflow.sklearn.log_model(model, "rf_model")


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest model")
    parser.add_argument(
        "data_path", type=str, help="Path to the training data CSV file"
    )
    args = parser.parse_args()
    train_model(data_path=args.data_path)


if __name__ == "__main__":
    main()
