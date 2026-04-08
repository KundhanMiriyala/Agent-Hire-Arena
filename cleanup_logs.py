import os
import shutil
import subprocess

os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')

# Remove logs directory
if os.path.exists('logs'):
    shutil.rmtree('logs')
    print("Removed logs directory")

# Git operations
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'Remove hardcoded logs and example output'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
print("Pushed to GitHub")
