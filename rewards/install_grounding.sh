#!/bin/bash

echo "=== Setting Up Grounding DINO ==="
cd "$(dirname "$0")/.."

# Clone or navigate to GroundingDINO
if [ ! -d "GroundingDINO" ]; then
    echo "Cloning GroundingDINO repository..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
fi

echo "=== Downloading Pre-trained Weights ==="
cd GroundingDINO
mkdir -p weights
cd weights
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi
