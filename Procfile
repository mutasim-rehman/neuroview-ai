web: cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 600 --worker-class sync --log-level info --access-logfile - --error-logfile - --timeout-keepalive 600 api_server:app

