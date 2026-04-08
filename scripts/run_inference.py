import sys
import os
from io import StringIO

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import main

class DualWriter:
    """Write to both stdout and a file simultaneously"""
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
    
    def flush(self):
        for stream in self.streams:
            if hasattr(stream, 'flush'):
                stream.flush()

if __name__ == '__main__':
    # Get absolute path to logs directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(repo_root, 'logs', 'real_baseline_run.txt')
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Open log file for writing
    log_file = open(log_path, 'w')
    
    # Redirect stdout to write to both console and file
    sys.stdout = DualWriter(sys.__stdout__, log_file)
    
    try:
        main()
    finally:
        if log_file:
            log_file.close()
        sys.stdout = sys.__stdout__
