#!/bin/bash
# Activation script for VQVAEAudioGenerator virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run setup.sh first to create the virtual environment."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Virtual environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

