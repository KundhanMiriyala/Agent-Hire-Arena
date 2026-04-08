import os
import subprocess

os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')

# Remove this cleanup script
if os.path.exists('cleanup_final.py'):
    os.remove('cleanup_final.py')
    print("Removed cleanup_final.py")

# Final commit
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'Remove cleanup script'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
print("Repository cleaned and ready for submission")
