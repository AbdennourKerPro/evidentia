# Start from a small official Python image.
FROM python:3.12-slim

# Keep Python logs visible in Docker and avoid creating .pyc files in the image.
# OpenVINO GenAI's native extension needs the runtime libraries bundled by the
# Python `openvino` wheel to be visible to Linux's linker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/openvino/libs

# All following paths are relative to /app inside the image.
WORKDIR /app

# Docling uses OpenCV for document-layout and table analysis. These runtime
# libraries are absent from the minimal Python image but required by OpenCV.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first so Docker can reuse this layer when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source after the dependency layer.
COPY app ./app

# The one-time model downloader uses the same image and writes to model_cache.
COPY scripts ./scripts

# Documentation metadata: the actual host-to-container mapping is set at runtime.
EXPOSE 8000

# Start the FastAPI application when the container starts.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
