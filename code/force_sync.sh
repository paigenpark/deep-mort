#!/bin/bash

# Real path to data folder inside Google Drive
REAL_PATH="$HOME/Library/CloudStorage/GoogleDrive-paige_park@berkeley.edu/My Drive/deep-mortality-data/data"

# Recursively update the modification time (without actually changing content)
find "$REAL_PATH" -type f -exec touch {} +

echo "Touched all files in $REAL_PATH to force Google Drive sync."

