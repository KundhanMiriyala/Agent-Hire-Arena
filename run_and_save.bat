@echo off
cd /d "c:\Users\munna\GitClones\Agent Hire-Arena"
set MOCK_OPENAI=1
.venv\Scripts\python inference.py > logs\real_baseline_run.txt 2>&1
echo Done. Output saved to logs\real_baseline_run.txt
