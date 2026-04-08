import os, subprocess
os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')
for f in ['sync_to_hf.py', 'sync.bat', 'test_openenv_push.py', 'push_openenv.py']:
    if os.path.exists(f): os.remove(f); print(f"Removed {f}")
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'Remove temp scripts'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
os.remove('cleanup_and_push.py')
print("Done!")
