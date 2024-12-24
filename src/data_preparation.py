import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prepare_data(file_path):
    with mlflow.start_run(run_name="Data Preparation", nested=True):
        # Log the file path as a parameter
        mlflow.log_param("data_source", file_path)
        
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Log basic dataset stats
        mlflow.log_param("rows", len(df))
        mlflow.log_param("columns", len(df.columns))
        
        # One-hot encode 'ocean_proximity'
        df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
        
        # Features and target variable
        X = df_encoded.drop(columns=["median_house_value"])
        y = df_encoded["median_house_value"]
        
        # Impute missing values
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        
        # Log split sizes
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        return X_train, X_test, y_train, y_test
