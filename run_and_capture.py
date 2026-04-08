import os
import sys

# Set mock mode
os.environ['MOCK_OPENAI'] = '1'
os.environ['API_BASE_URL'] = 'http://127.0.0.1:7860'
os.environ['MODEL_NAME'] = 'mock'

# Redirect output to capture it
from io import StringIO

output = StringIO()

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()

# Import after setting env vars
import inference

# Capture output
log_file = open('logs/real_baseline_run.txt', 'w')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

try:
    inference.main()
finally:
    sys.stdout = original_stdout
    log_file.close()
    print("\n✓ Full output saved to logs/real_baseline_run.txt")
