#!/bin/bash
cd "C:\Users\munna\GitClones\Agent Hire-Arena"
rm -rf logs
git add -A
git commit -m "Remove hardcoded logs and example output section from README"
git push origin main
