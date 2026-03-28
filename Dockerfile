# Use an official lightweight Python 3.11 image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install necessary system dependencies (useful for python packages building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache them in docker layer
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set default port fallback if the $PORT environment variable isn't passed (Render handles this dynamically)
ENV PORT=10000

# Expose the port to allow external connections
EXPOSE $PORT

# Start command explicitly bound to the network and port mapping
CMD uvicorn app.ui_main:app --host 0.0.0.0 --port $PORT
