web: cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --worker-class sync --log-level info --access-logfile - --error-logfile - --preload api_server:app

