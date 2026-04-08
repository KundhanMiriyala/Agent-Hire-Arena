import os,subprocess
os.chdir(r'C:\Users\munna\GitClones\Agent Hire-Arena')
for f in ['.env','cleanup.py','do_push.py','push_hf.py','push_hf.ps1']:
    if os.path.exists(f):os.remove(f)
subprocess.run(['git','add','-A'],check=True)
subprocess.run(['git','commit','-m','Remove secrets and temp scripts'],check=True)
subprocess.run(['git','push','origin','main'],check=True)
os.remove(__file__)
