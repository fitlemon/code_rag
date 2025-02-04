#!/bin/bash


# Start Ollama in the background
ollama serve &


# Wait for Ollama to start up
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags >/dev/null; do
    sleep 1
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "Ollama failed to start within 30 seconds. Exiting."
        exit 1
    fi
done

echo "Ollama is ready."
ollama pull bge-m3
ollama pull llama3.1
# Print the API URL
echo "API is running on: http://0.0.0.0:7860"

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 4 --limit-concurrency 20