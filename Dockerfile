# Multi-stage Docker build for ML Optimization Framework with Optuna Dashboard
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/home/mlopt/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlopt
WORKDIR /home/mlopt/app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies as root first
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip cache purge

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p logs results plots studies data && \
    chown -R mlopt:mlopt /home/mlopt/app

# Copy environment file
COPY .env.example .env

# Switch to non-root user
USER mlopt

# Expose ports for dashboard, jupyter, and streamlit
EXPOSE 8080 8888 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.data.data_pipeline; print('Framework OK')" || exit 1

# Default command - run comprehensive demo
CMD ["python", "scripts/deploy_complete_demo.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --user jupyter notebook ipywidgets

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter by default in dev mode
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --from=base --chown=mlopt:mlopt /home/mlopt/app /home/mlopt/app

# Set production environment
ENV ENVIRONMENT=production

# Start optimization service
CMD ["python", "scripts/cli_runner.py", "--model", "all", "--n_trials", "100", "--save_results"]
