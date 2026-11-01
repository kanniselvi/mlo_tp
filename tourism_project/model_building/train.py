import os
import pandas as pd
import joblib
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, classification_report

def main():
    # Ensure the deployment directory exists
    os.makedirs('tourism_project/deployment', exist_ok=True)

    # 1. Load the train and test splits produced by the previous step
    print("Loading train and test datasets...")
    train_df = pd.read_csv('tourism_project/data/train.csv')
    test_df = pd.read_csv('tourism_project/data/test.csv')

    # Separate target variable (ProdTaken)
    target_col = 'ProdTaken'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Identify numerical and categorical features
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # 2. Build preprocessing pipeline
    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown='ignore'), cat_cols)
    )

    # Define the Model Pipeline
    pipeline = make_pipeline(
        preprocessor,
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    )

    # 3. Define the parameter grid for tuning
    param_grid = {
        'xgbclassifier__max_depth': [3, 5],
        'xgbclassifier__n_estimators': [50, 100],
        'xgbclassifier__learning_rate': [0.1, 0.2]
    }

    # Enable MLflow autologging
    mlflow.set_experiment("Tourism_Package_Purchase_Prediction")
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        print("Tuning hyperparameters with GridSearchCV...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters: {best_params}")

        # 4. Evaluate the best model on the test set
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Log metrics to MLflow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_recall", recall)

        # 5. Save the best model locally in the deployment folder
        model_output_path = 'tourism_project/deployment/best_model.pkl'
        joblib.dump(best_model, model_output_path)
        print(f"Saved the best model successfully to: {model_output_path}")

if __name__ == '__main__':
    main()