FROM python:3.10-slim

WORKDIR /app

# Copy source code
COPY src/ .

# Install git and dependencies in requirements.txt (Also clean up to reduce image size)
RUN apt-get update && \
    apt-get install -y git && \
    pip install -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*





