from fastapi import FastAPI
import numpy as np
import mlflow.sklearn
import uvicorn
from src.mlflow_implementation.api.api_base.base import Item

app = FastAPI()

# Load the model and scaler from MLflow
model = mlflow.sklearn.load_model("mlruns/0/731267a6eb6341b58952a7cc05ca1d42/artifacts/model/MLmodel")
scaler = mlflow.sklearn.load_model("mlruns/0/731267a6eb6341b58952a7cc05ca1d42/artifacts/scaler/MLmodel")


@app.post("/predict")
async def predict(item: Item):
    """

    :param item:
    :return:
    """
    scaled_features = scaler.transform(np.array(item.features).reshape(1, -1))

    prediction = model.predict(scaled_features)[0]

    return {"prediction": prediction}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
