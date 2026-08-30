#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MEGA_URL="https://mega.nz/file/TB8FUb7D#D7REnXhbJbd8mR6KkcJbdOkIGHCuY3mJkxQjR_39-2o"
IMAGES_DIR="./IMAGES"
ZIP_FILE="$IMAGES_DIR/IMAGES.zip"

mkdir -p "$IMAGES_DIR"

# Check if zip already exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Choose download method:"
    echo "  1) Automatic (megatools) - may hit rate limits, run this script in tmux so that it runs in the background and can wait until rate limit resets"
    echo "  2) Manual - download yourself, script will provide you with all instructions step by step"
    read -p "Enter choice [1/2]: " choice

    if [ "$choice" = "1" ]; then
        if ! command -v megadl &> /dev/null; then
            echo "Installing megatools..."
            sudo apt update && sudo apt install -y megatools
        fi
        echo "Downloading (this may take a while due to rate limits)..."
        megadl --path "$IMAGES_DIR" "$MEGA_URL"
        # Rename to known filename
        mv "$IMAGES_DIR"/*.zip "$ZIP_FILE" 2>/dev/null || true
    else
        echo ""
        echo "Download the file manually from:"
        echo "  $MEGA_URL"
        echo ""
        echo "Save it as:"
        echo "  $SCRIPT_DIR/IMAGES/IMAGES.zip"
        echo ""
        read -p "Press Enter once the file is in place..."

        if [ ! -f "$ZIP_FILE" ]; then
            echo "Error: $ZIP_FILE not found"
            exit 1
        fi
    fi
fi

# Unzip images
echo "Unzipping images..."
cd "$IMAGES_DIR"
unzip -o IMAGES.zip
rm -f IMAGES.zip
cd "$SCRIPT_DIR"

# Unzip text passages
echo "Unzipping text passages..."
cd HybridQA
unzip -o text_passages.zip
cd "$SCRIPT_DIR"

echo "Done!"
