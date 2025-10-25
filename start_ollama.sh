#!/bin/bash

# Script to check and start Ollama for offline mode

echo " Checking Ollama installation..."

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo " Ollama is not installed."
    echo ""
    echo " To install Ollama:"
    echo "   macOS: brew install ollama"
    echo "   Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "   Or visit: https://ollama.com/download"
    exit 1
fi

echo " Ollama is installed"
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo " Ollama is not running. Starting Ollama..."
    echo ""
    
    # Start Ollama in the background
    ollama serve > /dev/null 2>&1 &
    
    # Wait for Ollama to start
    echo " Waiting for Ollama to start..."
    sleep 3
    
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo " Ollama started successfully"
    else
        echo " Failed to start Ollama"
        exit 1
    fi
else
    echo " Ollama is already running"
fi

echo ""
echo " Checking for llama3.2 model..."

# Check if llama3.2 model is available
if ollama list | grep -q "llama3.2"; then
    echo " llama3.2 model is available"
else
    echo "  llama3.2 model not found"
    echo ""
    echo " Pulling llama3.2 model (this may take a few minutes)..."
    ollama pull llama3.2:latest
    
    if [ $? -eq 0 ]; then
        echo " llama3.2 model downloaded successfully"
    else
        echo " Failed to download llama3.2 model"
        exit 1
    fi
fi

echo ""
echo " Ollama is ready for offline mode!"
echo ""
echo "Available models:"
ollama list
echo ""
echo "To run the agent in offline mode:"
echo "  export AGENT_MODE=offline"


