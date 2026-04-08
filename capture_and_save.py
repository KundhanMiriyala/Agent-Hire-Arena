#!/usr/bin/env python3
"""Run inference and save output to logs"""
import subprocess
import sys
import os

os.chdir(r'c:\Users\munna\GitClones\Agent Hire-Arena')
os.environ['MOCK_OPENAI'] = '1'

# Run inference.py and capture output
result = subprocess.run(
    [sys.executable, 'inference.py'],
    capture_output=True,
    text=True,
    timeout=120
)

# Write to logs file
with open('logs/real_baseline_run.txt', 'w') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write('\n\n=== STDERR ===\n')
        f.write(result.stderr)

# Also print to console
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

sys.exit(result.returncode)
