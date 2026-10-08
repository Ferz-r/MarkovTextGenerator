import logging.config
from copy import deepcopy

from uvicorn.config import LOGGING_CONFIG


def configure_logging() -> None:
    config = deepcopy(LOGGING_CONFIG)
    config["loggers"]["markov"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    logging.config.dictConfig(config)
