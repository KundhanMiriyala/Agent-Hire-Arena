import os
import subprocess

os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')

# Files to remove
to_remove = [
    'run_inference_mock.ps1',
    '.env',
]

for f in to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")

# Commit changes
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'Remove unnecessary temporary files and .env'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
print("Done - pushed to GitHub")
