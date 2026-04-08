# Push to GitHub
$github_user = "KundhanMiriyala"
$repo_name = "Agent-Hire-Arena"

Write-Host "Checking git status..."
git status

Write-Host "`nAdding all changes..."
git add -A

Write-Host "`nCommitting..."
git commit -m "OpenEnv submission: Agent Hire Arena environment with deterministic baseline"

Write-Host "`nChecking remote..."
git remote -v

$remote_exists = git remote | findstr "origin"
if (-not $remote_exists) {
    Write-Host "`nAdding GitHub remote..."
    git remote add origin "https://github.com/$github_user/$repo_name.git"
}

Write-Host "`nSetting main branch..."
git branch -M main

Write-Host "`nPushing to GitHub..."
git push -u origin main

Write-Host "`nDone! Your repo is now at: https://github.com/$github_user/$repo_name"
