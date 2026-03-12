##############################################################################
# VoiceBot - Complete Automated Deployment Script
# This script: authenticates GitHub, pushes code, and tells you the Render URL.
##############################################################################
$ErrorActionPreference = "Continue"
$host.UI.RawUI.WindowTitle = "VoiceBot Deployment"

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "  VoiceBot Full Deployment Pipeline" -ForegroundColor Cyan
Write-Host "======================================`n" -ForegroundColor Cyan

# ── Step 0: Check gh is installed ──
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: gh CLI not found. Install from https://cli.github.com" -ForegroundColor Red
    pause; exit 1
}

# ── Step 1: GitHub Auth ──
Write-Host "[1/5] Authenticating with GitHub..." -ForegroundColor Yellow
$authStatus = gh auth status 2>&1 | Out-String
if ($authStatus -match "Logged in") {
    Write-Host "  Already authenticated!" -ForegroundColor Green
} else {
    Write-Host "  A browser window will open. Click 'Authorize' and come back." -ForegroundColor White
    Write-Host ""
    gh auth login --hostname github.com --git-protocol https --web
    Start-Sleep 2
    $authCheck = gh auth status 2>&1 | Out-String
    if ($authCheck -notmatch "Logged in") {
        Write-Host "  ERROR: Authentication failed. Please try again." -ForegroundColor Red
        pause; exit 1
    }
    Write-Host "  Authenticated successfully!" -ForegroundColor Green
}

# ── Step 2: Initialize git repo ──
Write-Host "`n[2/5] Setting up git repository..." -ForegroundColor Yellow
$backendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendDir

# Create .gitignore
@"
__pycache__/
*.pyc
.env
*.egg-info/
venv/
.venv/
temp/
logs/*.log
models/intent_classifier/
audio_samples/*.wav
audio_samples/*.mp3
.idea/
.vscode/
*.swp
"@ | Out-File -Encoding utf8 "$backendDir\.gitignore" -Force

if (-not (Test-Path "$backendDir\.git")) {
    git init
}
git add -A
git commit -m "VoiceBot: AI Customer Support - full backend with ASR, NLP, TTS" --allow-empty 2>$null

Write-Host "  Git repo ready." -ForegroundColor Green

# ── Step 3: Create GitHub repo and push ──
Write-Host "`n[3/5] Creating GitHub repo and pushing..." -ForegroundColor Yellow
$repoName = "voicebot-backend"

# Check if repo exists
$existing = gh repo view $repoName 2>&1 | Out-String
if ($existing -match "could not resolve") {
    gh repo create $repoName --public --description "AI Voice Bot for Customer Support - FastAPI + Whisper + BERT + gTTS" --source . --push --remote origin
} else {
    Write-Host "  Repo already exists. Pushing update..." -ForegroundColor White
    git remote remove origin 2>$null
    $ghUser = gh api user --jq .login 2>$null
    git remote add origin "https://github.com/$ghUser/$repoName.git"
    git branch -M main
    git push -u origin main --force
}

Write-Host "  Code pushed to GitHub!" -ForegroundColor Green

# ── Step 4: Get the repo URL ──
$ghUser = gh api user --jq .login 2>$null
$repoUrl = "https://github.com/$ghUser/$repoName"
Write-Host "  Repo: $repoUrl" -ForegroundColor Cyan

# ── Step 5: Show Render deploy instructions ──
Write-Host "`n[4/5] Backend code is on GitHub!" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Your GitHub repo: $repoUrl" -ForegroundColor Cyan
Write-Host ""

# Create render.yaml in the repo
@"
services:
  - type: web
    name: voicebot-api
    runtime: docker
    plan: free
    envVars:
      - key: ASR_MODEL
        value: tiny
      - key: APP_PORT
        value: "10000"
"@ | Out-File -Encoding utf8 "$backendDir\render.yaml" -Force

git add render.yaml
git commit -m "Add render.yaml for deployment" 2>$null
git push origin main 2>$null

$renderUrl = "https://render.com/deploy?repo=$repoUrl"
Write-Host "  Click this link to deploy on Render (one-click):" -ForegroundColor Yellow
Write-Host "  $renderUrl" -ForegroundColor Green
Write-Host ""

# Auto-open the Render deploy page
Start-Process $renderUrl

Write-Host "`n[5/5] DONE!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  GitHub Repo : $repoUrl" -ForegroundColor White
Write-Host "  Deploy Link : $renderUrl" -ForegroundColor White
Write-Host "======================================`n" -ForegroundColor Cyan

# Save the repo URL for the frontend update
"$ghUser" | Out-File -Encoding utf8 "$backendDir\..\.gh_user.txt" -Force

Write-Host "Press any key to close..." -ForegroundColor Gray
pause
