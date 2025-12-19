#!/bin/bash
# Startup script for Render deployment

# Load model and start server
echo "Starting NeuroView AI API Server..."
echo "Loading model..."

# Start gunicorn with appropriate workers
exec gunicorn \
  --bind 0.0.0.0:${PORT:-5000} \
  --workers 2 \
  --timeout 120 \
  --worker-class sync \
  --access-logfile - \
  --error-logfile - \
  api_server:app

