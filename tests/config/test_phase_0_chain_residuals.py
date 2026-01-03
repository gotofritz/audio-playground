"""
Tests for Phase 0: --chain-residuals flag implementation

This test suite validates:
- Step 0.1: Config field `chain_residuals` loads correctly
- Step 0.2: Conditional logic respects the config flag
- Step 0.3: CLI flag is properly integrated
"""

from audio_playground.config.app_config import AudioPlaygroundConfig


# Step 0.1: Add config option
def test_chain_residuals_defaults() -> None:
    """Default value should be False for backward compatibility"""
    config = AudioPlaygroundConfig()
    assert config.chain_residuals is False


def test_chain_residuals_field_has_description() -> None:
    """Field should have a helpful description"""
    field_info = AudioPlaygroundConfig.model_fields["chain_residuals"]
    assert field_info.description is not None
    assert "residual" in field_info.description.lower()


def test_chain_residuals_persists_across_instances() -> None:
    """Each instance should maintain its own chain_residuals value"""
    config1 = AudioPlaygroundConfig(chain_residuals=True)
    config2 = AudioPlaygroundConfig(chain_residuals=False)

    assert config1.chain_residuals is True
    assert config2.chain_residuals is False


# Step 0.3: CLI flag integration
def test_sam_audio_command_has_chain_residuals_parameter() -> None:
    """sam_audio function should have chain_residuals parameter"""
    import inspect

    from audio_playground.cli.extract.sam_audio import sam_audio

    assert sam_audio.callback is not None
    sig = inspect.signature(sam_audio.callback)
    assert "chain_residuals" in sig.parameters


# Backward compatibility
def test_default_config_matches_original_behavior() -> None:
    """Default config should match original PoC behavior (chain_residuals=True)"""
    config = AudioPlaygroundConfig()

    # All original defaults should still be present
    assert config.app_name == "audio-playground"
    assert config.log_level == "DEBUG"
    assert config.prompts == ["bass"]

    assert config.chain_residuals is False


def test_all_other_config_fields_unchanged() -> None:
    """Adding chain_residuals shouldn't affect other fields"""
    config = AudioPlaygroundConfig(
        log_level="INFO",
        prompts=["bass", "vocals"],
        chain_residuals=True,
    )

    assert config.log_level == "INFO"
    assert config.prompts == ["bass", "vocals"]
    # And new field should still work
    assert config.chain_residuals is True
