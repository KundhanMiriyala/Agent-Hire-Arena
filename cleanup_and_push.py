import os
import subprocess

os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')

# Remove temporary files
files_to_remove = [
    'sync_to_hf.py',
    'sync.bat',
    'test_openenv_push.py',
    'push_openenv.py',
]

for f in files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"✓ Removed {f}")

# Commit to GitHub
print("\nCommitting to GitHub...")
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'Remove temporary sync scripts'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)

print("\n✅ Repository cleaned and pushed to GitHub")

# Remove this script itself
os.remove(__file__)
