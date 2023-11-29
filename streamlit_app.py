import streamlit as st
import requests
from src.mlflow_implementation.api.api_base import logger_sys


def main():
    """
    Streamlit UI and API connection
    """
    st.title('MLflow Implementation')

    label1 = st.text_input('Label 1', key='label1')
    label2 = st.text_input('Label 2', key='label2')
    label3 = st.text_input('Label 3', key='label3')
    label4 = st.text_input('Label 4', key='label4')

    if st.button("Predict"):
        try:
            features = [float(label1), float(label2), float(label3), float(label4)]

            data = {
                "feature1": features[0],
                "feature2": features[1],
                "feature3": features[2],
                "feature4": features[3]
            }
            logger_sys.info("-------------Streamlit server Started----------")

            response = requests.post("http://localhost:8000/predict", json=data)

            if response.status_code == 200:
                prediction = response.json().get("prediction")
                if prediction is not None:
                    st.write(f"Prediction: {prediction}")
                else:
                    st.error("Prediction not received from the server")
            else:
                st.error(f"Failed to get prediction from server: {response.text}")
                logger_sys.error("Failed to get prediction from server. Status code: %s", response.status_code)

        except requests.RequestException as e:
            st.error(f"Request to server failed: {str(e)}")
            logger_sys.error("Request to server failed: %s", str(e))

        except ValueError as e:
            st.error(f"Invalid input. Please enter numeric values: {str(e)}")
            logger_sys.error("Invalid input. Please enter numeric values: %s", str(e))


if __name__ == "__main__":
    main()
