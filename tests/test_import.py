"""Test audio_playground."""

import audio_playground


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(audio_playground.__name__, str)
