import logging
import logging.handlers
import multiprocessing
import datetime
from pathlib import Path
import atexit
from typing import Tuple, Optional

DEFAULT_LOGS_DIR = "logs"


def setup_logging(project_name: str, logs_dir: str = DEFAULT_LOGS_DIR, level: int = logging.INFO,
                  queue_logging: bool = True) -> Tuple[logging.Logger, Optional[logging.handlers.QueueListener]]:
    """
    Configure parallel-safe logging.

    Creates a log file named ``{project_name}-{timestamp}.log`` inside ``logs_dir``.
    If ``queue_logging`` is True, uses ``multiprocessing.Queue`` with ``QueueHandler``/``QueueListener``
    for process-safe logging. Returns the application logger and the listener (if any).
    Caller does not need to manage handlers individually; the listener is auto-stopped at exit.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    logfile = logs_path / f"{project_name}-{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger("bidsifier")
    logger.setLevel(level)

    listener: Optional[logging.handlers.QueueListener] = None

    if queue_logging:
        log_queue: multiprocessing.Queue = multiprocessing.Queue(-1)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        file_handler = logging.FileHandler(str(logfile), encoding="utf-8")
        file_handler.setFormatter(formatter)
        listener = logging.handlers.QueueListener(log_queue, file_handler)
        listener.start()
        logger.addHandler(queue_handler)

        def _stop_listener() -> None:
            try:
                listener.stop()
            except Exception:
                pass
        atexit.register(_stop_listener)
    else:
        file_handler = logging.FileHandler(str(logfile), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Also add a simple stderr stream handler for immediate feedback.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Logging initialized: %s", logfile)
    return logger, listener
