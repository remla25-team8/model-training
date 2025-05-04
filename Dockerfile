FROM python:3.10-slim

WORKDIR /app

# Copy source code
COPY src/ .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt





