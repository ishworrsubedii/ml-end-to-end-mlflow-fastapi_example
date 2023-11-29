import numpy as np
from fastapi import FastAPI, HTTPException
from src.mlflow_implementation.api.api_base import logger_sys
from src.mlflow_implementation.api.api_base.base import Item
from src.mlflow_implementation.services.example_service import ExampleService

app = FastAPI()


@app.post("/predict")
async def data_fetch(item: Item):
    try:
        features = np.array([[item.feature1, item.feature2, item.feature3, item.feature4]])
        obj_example = ExampleService()
        scaled_features, prediction = obj_example.main_prediction(features)
        prediction = prediction.tolist()

        logger_sys.info(f"Prediction successful for features: {features}, Prediction: {prediction}")
        return {"prediction": prediction}

    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        logger_sys.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
