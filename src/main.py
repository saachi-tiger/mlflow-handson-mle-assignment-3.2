import mlflow
from data_preparation import prepare_data
from model_training import train_model
from model_scoring import score_model

if __name__ == "__main__":
    # Start a parent run
    with mlflow.start_run(run_name="Housing Pipeline"):
        # File path to the dataset
        file_path = "../notebooks/datasets/housing/housing.csv"
        
        # Step 1: Data Preparation
        X_train, X_test, y_train, y_test = prepare_data(file_path)
        
        # Step 2: Model Training
        model = train_model(X_train, y_train, X_test, y_test)
        
        # Step 3: Model Scoring
        score_model(model, X_test, y_test)
