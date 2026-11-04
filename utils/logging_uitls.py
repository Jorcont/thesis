# import logging
# import sys
# from typing import Literal

# logger = logging.getLogger("meta-netze")

# def setup_logger(logging_level: int | Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
#     logging.basicConfig(level=logging_level)
#     logger.setLevel(logging_level)

#     formatter = logging.Formatter(
#         fmt="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
#     )

#     # DEBUG, INFO → stdout → SLURM .out file
#     out_handler = logging.StreamHandler(sys.stdout)
#     out_handler.setLevel(logging.DEBUG)
#     out_handler.addFilter(lambda record: record.levelno < logging.WARNING)
#     out_handler.setFormatter(formatter)

#     # WARNING, ERROR, CRITICAL → stderr → SLURM .err file
#     err_handler = logging.StreamHandler(sys.stderr)
#     err_handler.setLevel(logging.WARNING)
#     err_handler.setFormatter(formatter)

#     logger.addHandler(out_handler)
#     logger.addHandler(err_handler)

import sys
import logging
from typing import Literal

def setup_logger(logging_level: int | Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
    
    # Alles → stderr → SLURM .err file
    err_handler = logging.StreamHandler(sys.stderr)   # ← war sys.stdout
    err_handler.setLevel(logging.DEBUG)
    err_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger = logging.getLogger("meta-netze")
    logger.handlers.clear()          # alte Handler entfernen
    logger.addHandler(err_handler)
    logger.setLevel(logging_level)
    logger.propagate = False
    return logger