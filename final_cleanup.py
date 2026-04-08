import os
os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')

# Remove temporary scripts
for f in ['capture_and_save.py', 'capture_logs.py', 'cleanup.bat', 'cleanup.sh', 'cleanup_logs.py', 'run_and_capture.py', 'run_and_save.bat', 'run_baseline.ps1']:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")

import subprocess
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'Remove temporary cleanup scripts'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
print("Done")
