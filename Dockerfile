FROM python:3.10.13

RUN apt update -y

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "inference.py","streamlit_app.py"]
