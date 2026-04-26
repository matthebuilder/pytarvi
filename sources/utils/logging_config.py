import sys
from loguru import logger


def setup_logging() -> None:
    """
    Configures loguru for the application.
    Logs are sent to both stdout and a file.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add("app.log", rotation="10 MB", retention="10 days", level="DEBUG")


if __name__ == "__main__":
    setup_logging()
    logger.info("Logging configured successfully.")
