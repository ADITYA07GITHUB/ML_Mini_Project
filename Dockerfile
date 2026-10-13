# NetGuard AI - Production & Evaluation Dockerfile
FROM python:3.12-slim

# Prevent Python from writing .pyc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential for C extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and directories
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY ml/ ./ml/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/ ./data/

# Generate demonstration dataset and pre-train models so container is immediately operational
RUN python scripts/generate_demo_data.py --count 4000
RUN python -m ml.train --data data/demo/cicids2017_demo.csv --mode binary
RUN python -m ml.train --data data/demo/cicids2017_demo.csv --mode multiclass

# Expose FastAPI server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Launch application server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
