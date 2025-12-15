FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create output directories
RUN mkdir -p data/raw data/processed models reports/eda reports/anomalies reports/predictions output

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command - run the full pipeline
CMD ["python", "src/main_pipeline.py"]
