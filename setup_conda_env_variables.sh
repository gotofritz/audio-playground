#!/bin/bash

# Ensure CONDA_PREFIX is set (must be in an active environment)
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No active Conda environment detected. Please 'conda activate' your env first."
    exit 1
fi

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

# ---------------------------------------------------------
# 1. Create Activation Script
# ---------------------------------------------------------
cat <<EOF > "$ACTIVATE_DIR/env_vars.sh"
#!/bin/sh

# Save existing library path to restore on deactivation
export _OLD_DYLD_FALLBACK_LIBRARY_PATH="\$DYLD_FALLBACK_LIBRARY_PATH"

# Dynamic path for macOS FFmpeg linking
export DYLD_FALLBACK_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$DYLD_FALLBACK_LIBRARY_PATH"

# Offline and Pip variables
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_REQUIRE_VIRTUALENV=false
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

echo "✓ sam-audio environment variables set."
EOF

# ---------------------------------------------------------
# 2. Create Deactivation Script
# ---------------------------------------------------------
cat <<EOF > "$DEACTIVATE_DIR/env_vars.sh"
#!/bin/sh

# Restore original library path
export DYLD_FALLBACK_LIBRARY_PATH="\$_OLD_DYLD_FALLBACK_LIBRARY_PATH"
unset _OLD_DYLD_FALLBACK_LIBRARY_PATH

# Unset custom variables
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset PIP_REQUIRE_VIRTUALENV
unset TOKENIZERS_PARALLELISM

echo "✓ sam-audio environment variables cleared."
EOF

echo "Setup complete. Please 'conda deactivate' and 'conda activate' to apply changes."
