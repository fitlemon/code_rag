FROM python:3.11-slim

# Install curl and Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.ai/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up user and environment
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt


COPY --chown=user . .

# Make the start script executable
RUN chmod +x start.sh

CMD ["./start.sh"]