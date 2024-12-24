import mlflow
from sklearn.metrics import mean_squared_error, r2_score

def score_model(model, X_test, y_test):
    with mlflow.start_run(run_name="Model Scoring", nested=True):
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
