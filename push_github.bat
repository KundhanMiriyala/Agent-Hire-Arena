@echo off
cd /d "C:\Users\munna\GitClones\Agent Hire-Arena"
echo Checking git status...
git status
echo.
echo Adding files...
git add -A
echo.
echo Committing...
git commit -m "OpenEnv: Agent Hire Arena with deterministic baseline and metrics"
echo.
echo Checking remote...
git remote -v
echo.
if errorlevel 128 (
    echo Adding GitHub remote...
    git remote add origin https://github.com/KundhanMiriyala/Agent-Hire-Arena.git
)
echo.
echo Setting main branch...
git branch -M main
echo.
echo Pushing to GitHub...
git push -u origin main
echo.
echo Done! Repo is at: https://github.com/KundhanMiriyala/Agent-Hire-Arena
pause
