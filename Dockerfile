# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

# Set timezone and non-interactive frontend for libsndfile/ffmpeg
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies required for handling audio (librosa/soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces strictly requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY --chown=user requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files (including your .pt models)
COPY --chown=user . /app

# Ensure correct permissions for temporary file generation during runs
RUN mkdir -p /tmp/health_screening_uploads/ && chmod 777 /tmp/health_screening_uploads/

# Expose port exactly as Hugging Face Spaces expects
EXPOSE 7860

# Command to run on startup
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
