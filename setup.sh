#!/bin/bash
# Setup script for VQWave project
# Creates a Python virtual environment and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "Setting up VQWave environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check for PortAudio (required for pyaudio)
if ! pkg-config --exists portaudio-2.0 2>/dev/null && [ ! -f /usr/include/portaudio.h ] && [ ! -f /usr/local/include/portaudio.h ]; then
    echo ""
    echo "Warning: PortAudio development libraries not found."
    echo "PyAudio requires PortAudio to be installed."
    echo ""
    echo "To install PortAudio, run one of the following:"
    echo "  Ubuntu/Debian: sudo apt-get install portaudio19-dev"
    echo "  Fedora/RHEL:   sudo dnf install portaudio-devel"
    echo "  Arch Linux:    sudo pacman -S portaudio"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode (installs all dependencies from setup.py)
echo "Installing package and dependencies in editable mode..."
pip install -e "$SCRIPT_DIR"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Or use the activate script:"
echo "  source activate.sh"

