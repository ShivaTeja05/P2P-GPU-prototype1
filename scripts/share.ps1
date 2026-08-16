<#
.SYNOPSIS
    Share this PC's GPU with a friend.

.DESCRIPTION
    Asks how long to share for, starts the container, and copies the connection
    URL to your clipboard so you can paste it straight into a chat.
#>
[CmdletBinding()]
param(
    [double]$Hours = 0,
    [string]$Gpu = 'all'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$P2pgpu = Join-Path $RepoRoot '.venv\Scripts\p2pgpu.exe'

if (-not (Test-Path $P2pgpu)) {
    Write-Host "p2pgpu is not installed yet. Run Setup.bat first." -ForegroundColor Red
    exit 1
}

Write-Host @"

  Share my GPU
  ------------

"@ -ForegroundColor White

# Refuse to start a second share on top of a running one.
& $P2pgpu url *>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  A share is already running." -ForegroundColor Yellow
    & $P2pgpu status
    Write-Host ""
    Write-Host "  Stop it first with Stop-Sharing.bat" -ForegroundColor Yellow
    exit 1
}

if ($Hours -le 0) {
    $answer = Read-Host "  How many hours? [4]"
    $Hours = if ([string]::IsNullOrWhiteSpace($answer)) { 4 } else { [double]$answer }
}

Write-Host ""
Write-Host "  Starting. The first run downloads a few GB of PyTorch image -" -ForegroundColor DarkGray
Write-Host "  that happens once, later shares start in seconds." -ForegroundColor DarkGray
Write-Host ""

& $P2pgpu share --hours $Hours --gpu $Gpu
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Could not start the share. Run Check-Setup.bat to find out why." -ForegroundColor Red
    exit 1
}

$url = & $P2pgpu url 2>$null
if ($LASTEXITCODE -eq 0 -and $url) {
    try {
        Set-Clipboard -Value $url
        Write-Host ""
        Write-Host "  URL copied to your clipboard - paste it to your friend." -ForegroundColor Green
    } catch {
        Write-Host ""
        Write-Host "  Send your friend this URL:" -ForegroundColor Green
        Write-Host "  $url" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "  Treat that URL like a password - anyone who has it can use the GPU." -ForegroundColor Yellow
Write-Host "  It stops automatically after $Hours hour(s), or run Stop-Sharing.bat." -ForegroundColor DarkGray
Write-Host ""
