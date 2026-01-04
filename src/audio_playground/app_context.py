from audio_playground import __app_name__
from audio_playground.config.app_config import AudioPlaygroundConfig
from audio_playground.logging.logging import setup_logger


class AppContext:
    """Holds all the objects needed by commands"""

    def __init__(self) -> None:
        self.app_config = AudioPlaygroundConfig(app_name=__app_name__)
        self.logger = setup_logger(log_level=self.app_config.log_level, app_name=__app_name__)
