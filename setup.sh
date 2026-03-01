#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

echo "Creating data directories..."
mkdir -p data/raw/claude data/raw/gemini data/raw/chatgpt data/processed data/outputs/plots

echo ""
echo "Setup complete."
echo "Drop your export files in data/raw/ and run: python main.py parse"
echo ""
echo "  Claude:  data/raw/claude/conversations.json"
echo "  Gemini:  data/raw/gemini/MyActivity.html"
echo "  ChatGPT: data/raw/chatgpt/conversations.json"
