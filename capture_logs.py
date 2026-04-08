#!/usr/bin/env python
"""Capture real inference output and save to logs"""
import subprocess
import sys
import os

os.environ['MOCK_OPENAI'] = '1'

result = subprocess.run(
    [sys.executable, 'inference.py'],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    capture_output=True,
    text=True
)

# Write output to logs
with open('logs/real_baseline_run.txt', 'w') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write('\n--- STDERR ---\n')
        f.write(result.stderr)

print(result.stdout)
if result.returncode != 0:
    print(f"Exit code: {result.returncode}")
    print("STDERR:", result.stderr)
