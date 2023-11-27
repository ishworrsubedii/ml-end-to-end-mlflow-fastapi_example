import mlflow
from src.mlflow_implementation import logger
from src.mlflow_implementation.pipeline.stage_1_data_creation import DataCreationPipeline
from src.mlflow_implementation.pipeline.stage_2_data_preprocessing import DataPreprocessPipeline
from src.mlflow_implementation.pipeline.stage_3_model_training import ModelTrainPipeline
from src.mlflow_implementation.pipeline.stage_4_model_eval import ModelEvalPipeline
from sklearn.metrics import accuracy_score


def stage_one():
    try:
        logger.info(">>------ State Data Creation Stage started -------<<")
        obj = DataCreationPipeline()
        X, y = obj.main()
        logger.info(">>------ State Data Creation Stage completed -------<<")
        return X, y
    except Exception as e:
        logger.exception(e)
        raise e


def stage_two(X, y):
    try:
        logger.info(">>------ State Data Preprocess Stage started -------<<")
        obj = DataPreprocessPipeline()
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = obj.main(X, y)
        logger.info(">>------ State Data Preprocess Stage completed -------<<")
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.exception(e)
        raise e


def stage_three(X_train_scaled, y_train):
    try:
        logger.info(">>------ State Model Training Stage started -------<<")
        obj = ModelTrainPipeline()
        model = obj.main(X_train_scaled, y_train)
        logger.info(">>------ State Model Training Stage completed -------<<")
        return model
    except Exception as e:
        logger.exception(e)
        raise e


def stage_four(y_test, model, X_test_scaled):
    try:
        logger.info(">>------ State Model Evaluation Stage started -------<<")
        obj = ModelEvalPipeline()
        accuracy_ = obj.main(y_test, model, X_test_scaled)
        logger.info(">>------ State Model Evaluation Stage completed -------<<")
        return accuracy_
    except Exception as e:
        logger.exception(e)
        raise e


def log_mlflow_run(model, X_test_scaled, y_test, scaler):
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test_scaled)))
        mlflow.sklearn.log_model(scaler, "scaler", registered_model_name="scaler")
        mlflow.sklearn.log_model(model, "model")


if __name__ == '__main__':
    try:
        X, y = stage_one()
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = stage_two(X, y)
        model = stage_three(X_train_scaled, y_train)
        accuracy = stage_four(y_test, model, X_test_scaled)
        log_mlflow_run(model, X_test_scaled, y_test, scaler)
    except Exception as e:
        logger.exception(e)
