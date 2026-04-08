import os, subprocess
os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')
for f in ['sync_to_hf.py', 'sync.bat', 'test_openenv_push.py', 'push_openenv.py', 'cleanup_and_push.py']:
    try:
        if os.path.exists(f):
            os.remove(f)
            print(f"✓ Removed {f}")
    except:
        pass
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'Remove temporary scripts'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
os.remove('final.py')
print("✅ Cleaned and pushed!")
