from audio_playground.config.app_config import AudioPlaygroundConfig


def test_settings() -> None:
    settings = AudioPlaygroundConfig()
    assert settings.app_name == "audio-playground"
    assert settings.log_level in ["INFO", "DEBUG"]
