import argparse
import os
import mlflow.sklearn
import pandas as pd

def evaluate(data_path: str, run_id: str) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        local_uri = "http://localhost:5000"
        print(f"MLflow tracking URI not set. Using local store at {local_uri}")
        mlflow.set_tracking_uri(local_uri)

    if not run_id:
        # Get the latest run ID from MLflow
        runs = mlflow.search_runs(
            filter_string='tags.type = "training"',
            order_by=["start_time DESC"]
        )
        run_id = runs.iloc[0].run_id
        
    model_uri = f"runs:/{run_id}/rf_model"

    model = mlflow.sklearn.load_model(model_uri)

    df_test = pd.read_csv(data_path)

    submission = pd.DataFrame({
        'Id': df_test['Id']
    })

    df_test.drop('Id', axis=1, inplace=True)

    predictions = model.predict(df_test)

    submission['Cover_Type'] = predictions

    submission.to_csv("result.csv", index=False)

    cover_type_mapping = {
        1: 'Spruce/Fir',
        2: 'Lodgepole Pine',
        3: 'Ponderosa Pine',
        4: 'Cottonwood/Willow',
        5: 'Aspen',
        6: 'Douglas-fir',
        7: 'Krummholz'
    }

    df = pd.read_csv('result.csv')

    df['cover_type_name'] = df['Cover_Type'].map(cover_type_mapping)

    df.to_csv('fullResult.csv', index=False)

    with mlflow.start_run(run_name="evaluate", nested=True):
        mlflow.log_param("model_run_id", run_id)

        # Log the predictions
        mlflow.log_artifact("result.csv")
        mlflow.log_artifact("fullResult.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RandomForest model"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the CSV file containing test feat and target",
    )
    parser.add_argument(
        "--run_id",
        required=False,
        default=None,
        type=str,
        help="Run_id of mlflow model to evaluate",
    )
    args = parser.parse_args()
    evaluate(data_path=args.data_path, run_id=args.run_id)


if __name__ == "__main__":
    main()
