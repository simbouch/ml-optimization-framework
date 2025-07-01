# Production Docker image for ML Optimization Framework with Optuna Integration
FROM python:3.11-slim

# Set metadata
LABEL maintainer="ML Optimization Framework Team"
LABEL description="Production-ready ML optimization framework with Optuna and Streamlit"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/simbouch/ml-optimization-framework"

# Set working directory
WORKDIR /app

# Install system dependencies (minimal but comprehensive)
RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements-minimal.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-minimal.txt

# Copy source code
COPY src/ ./src/

# Copy examples
COPY examples/ ./examples/

# Copy application files
COPY create_unified_demo.py .

# Create necessary directories with proper permissions
RUN mkdir -p studies logs results && \
    chmod 755 studies logs results

# Create non-root user with proper permissions
RUN useradd --create-home --shell /bin/bash mlopt && \
    chown -R mlopt:mlopt /app && \
    chmod +x *.py

# Switch to non-root user
USER mlopt

# Expose ports for Streamlit and Optuna Dashboard
EXPOSE 8501 8080

# Health check for Optuna Dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080 || exit 1

# Default command - run demo and start dashboard
CMD ["sh", "-c", "python create_unified_demo.py && optuna-dashboard sqlite:///studies/unified_demo.db --host 0.0.0.0 --port 8080"]
