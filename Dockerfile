# GraphRAG Application Container
# Python 3.10 + FAISS + BGE-M3 + Neo4j driver

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY docker-requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for Phase 3
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-mock==3.12.0 \
    python-dotenv==1.0.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    data/processed/faiss \
    data/interim/entities \
    logs

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "-m", "pytest", "tests/", "-v"]