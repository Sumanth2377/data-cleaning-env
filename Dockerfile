# ============================================================================
# Dockerfile — Data Cleaning OpenEnv
# ============================================================================
# Build:  docker build -t data-cleaning-env .
# Run:    docker run -p 7860:7860 data-cleaning-env
# ============================================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for HuggingFace Spaces compatibility
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY environment/ ./environment/
COPY server/      ./server/
COPY inference.py .
COPY openenv.yaml .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Health-check — ping /health every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
