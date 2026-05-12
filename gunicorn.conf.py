# gunicorn.conf.py
import os
import multiprocessing

# Bind to port
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker settings for memory efficiency
workers = 1
worker_class = 'sync'
threads = 2
worker_connections = 1000

# Timeout settings
timeout = 300
graceful_timeout = 60
keepalive = 5

# Memory management
max_requests = 200
max_requests_jitter = 50

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process name
proc_name = 'market-tracker'
