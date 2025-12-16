#!/bin/bash
# Script to run the Gradio application

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check if .env exists and has API key
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please copy env.example to .env and add your OpenAI API key."
    exit 1
fi

# Check if API key is set (not the placeholder)
if grep -q "your_openai_api_key_here" .env; then
    echo "Warning: Please add your OpenAI API key to the .env file"
    echo "Edit backend/.env and replace 'your_openai_api_key_here' with your actual API key"
    read -p "Press Enter to continue anyway (will fail if API key is needed) or Ctrl+C to exit..."
fi

# Run the Gradio application
echo "Starting Gradio application..."
echo "The application will be available at http://localhost:8000"
python gradio_main.py

