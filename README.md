# 🚀 audio-playground

![Coverage](https://raw.githubusercontent.com/gotofritz/audio-playground/badges/coverage.svg)

Playing with sam-audio

## 📋 Overview

`audio-playground` is a research environment for experimenting with Meta's SAM-Audio model. Due to specific hardware and compiler requirements on M1/M2/M3 Macs, this project uses **Conda** to manage C++ dependencies (FFmpeg) and **PyTorch Nightly** builds to ensure compatibility.

The best way of running this is is as a subfolder of a conda project, as described [in this blog post](https://gotofritz.net/blog/2025-12-20-playing-with-the-sam-audio-model-on-my-m1-macbook/)

## 🚀 Quick Start

### Prerequisites

- macOS (Apple Silicon)
- [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda installed
- [Task](https://taskfile.dev/) (optional, for automation)

### Installation & Environment Setup

Because this project relies on local patches for Meta's research repos, the setup must be done in a specific order:

1. **Clone the repository:**

```sh
git clone https://github.com/gotofritz/audio-playground.git
cd audio-playground
```

1. **Create and initialize the Conda environment:**

```sh
conda create -n sam-audio python=3.12 -y
conda activate sam-audio

# install the env variables
./setup_conda_env_variables.sh
conda activate sam-audio
```

1. **Install Core macOS Dependencies:**

```sh
# Install project-specific FFmpeg
conda install -c conda-forge ffmpeg=7.1 -y

# Install Nightly PyTorch & TorchCodec (Critical for M1/M2/M3)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall
pip install --pre torchcodec --index-url https://download.pytorch.org/whl/nightly/cpu

```

1. **Install Local & Research Dependencies:**
   Ensure you have your patched versions of `sam-audio` and `perception-models` available locally.

```sh
# Install research modules
pip install git+https://github.com/facebookresearch/ImageBind.git
pip install av --no-binary av

# Install this project in editable mode
pip install -e . --no-deps
```

## 🖥️ CLI Usage

`audio-playground` provides a CLI to run audio separation tasks:

```sh
# Show available commands
audio-playground --help

# Run the separation test suite
audio-playground test-run
```

Example usage:

```sh
❯ audio-playground test-run
2025-12-24 18:28:24 [INFO] Starting...
2025-12-24 18:28:24 [INFO] Using mps device
...
2025-12-24 18:28:30 [INFO] Done. Results saved to ./wav/processed/
```

### Development Setup

1. **Install pre-commit hooks:**

   ```sh
   pre-commit install
   ```

2. **Run the test suite:**

   ```sh
   task test
   ```

3. **Check code quality:**

   ```sh
   task qa
   ```

## 🛠️ Development

### Project Structure

```
audio-playground/
├── audio_playground/      # Main package
│   ├── cli/               # CLI commands and entry points
│   ├── config/            # Pydantic-settings configuration
│   └── ...                # Logic modules
├── tests/                 # Test suite
├── pyproject.toml         # Build system & metadata
└── README.md              # This file

```

### Common Commands

```sh
# Run tests
task test

# Linting and Type checking
task qa

# Set environment vars
./setup_conda_env_variables.sh
conda deactivate
conda activate sam-audio

# refresh the CLI tool
pip install --no-deps --no-cache-dir .
```

## 🧪 Testing

The project uses `pytest`. Ensure your environment is active:

```sh
pytest tests/

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🔗 Links

- **Main Project:** [https://github.com/gotofritz/audio-playground](https://github.com/gotofritz/audio-playground)
- **SAM-Audio Repo:** [facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)
