import logging

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once; repeated calls (Streamlit reruns) are no-ops."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT)
