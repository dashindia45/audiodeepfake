#!/bin/bash

echo "🚀 Setting up Audio Deepfake Detection Project..."

# Install dependencies
echo "📦 Installing requirements..."
python -m pip install -r requirements.txt

# Run pipeline
echo "▶️ Running pipeline..."
python src/run_pipeline.py

echo "🎉 Done!"