# Base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-ttf-dev \
        libsmpeg-dev \
        libportmidi-dev \
        libavformat-dev \
        libswscale-dev \
        libavcodec-dev \
        libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy application files and images
COPY main.py sam_vit_h_4b8939.pth ./
COPY images/ ./images/

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    pygame numpy pillow git+https://github.com/facebookresearch/segment-anything.git

# Default command
CMD ["python", "main.py"]