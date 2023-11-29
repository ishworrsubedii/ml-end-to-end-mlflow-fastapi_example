from configparser import ConfigParser
from src.mlflow_implementation.api.fastapi_app import app
from src.mlflow_implementation.api.api_base import logger_sys
import uvicorn

if __name__ == '__main__':
    logger_sys.info("-------------Fast Api server Started----------")

    configure = ConfigParser()
    configure.read('config/config.ini')

    local_host = configure.get("default", "host")
    port = configure.getint("default", "port")

    uvicorn.run(app, host=local_host, port=port)
