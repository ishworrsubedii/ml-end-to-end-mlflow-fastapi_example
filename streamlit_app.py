import streamlit as st
import requests
import uvicorn
from src.mlflow_implementation.api.fastapi_app import app


def streamlit_ui():
    # Streamlit UI
    st.title("FastAPI and Streamlit Integration")

    st.sidebar.header("Enter Feature Values:")
    features = []
    for i in range(20):
        features.append(st.sidebar.slider(f"Feature {i + 1}", float(0.0), float(1.0)))  # Update the range accordingly

    if st.sidebar.button("Predict"):
        payload = {"features": features}

        response = requests.post("http://localhost:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.write("## Prediction:")
            st.write(f"The model predicts the class: **{result['prediction']}**")
        else:
            st.error("Error in making prediction.")


if __name__ == '__main__':
    streamlit_ui()
