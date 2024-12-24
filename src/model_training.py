import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Model Training", nested=True):
        # Hyperparameters
        n_estimators = 100
        max_depth = None
        random_state = 42
        
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Log metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
