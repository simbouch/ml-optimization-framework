# Multi-stage Docker build for ML Optimization Framework
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlopt
WORKDIR /home/mlopt/app
USER mlopt

# Copy requirements first for better caching
COPY --chown=mlopt:mlopt requirements.txt .

# Install Python dependencies
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=mlopt:mlopt . .

# Fix permissions and create directories
RUN chown -R mlopt:mlopt /home/mlopt/app && \
    mkdir -p logs results plots studies

# Expose ports
EXPOSE 8080 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.data.data_pipeline; print('OK')" || exit 1

# Default command
CMD ["python", "scripts/run_optimization.py"]

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
