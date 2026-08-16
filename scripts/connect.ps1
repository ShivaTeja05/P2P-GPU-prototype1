<#
.SYNOPSIS
    Connect to a GPU your friend is sharing.

.DESCRIPTION
    Takes the URL your friend sent you, checks it is reachable, and opens the
    notebook in your browser.
#>
[CmdletBinding()]
param(
    [string]$Url = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$P2pgpu = Join-Path $RepoRoot '.venv\Scripts\p2pgpu.exe'

if (-not (Test-Path $P2pgpu)) {
    Write-Host "p2pgpu is not installed yet. Run Setup.bat first." -ForegroundColor Red
    exit 1
}

Write-Host @"

  Connect to a shared GPU
  -----------------------

"@ -ForegroundColor White

if ([string]::IsNullOrWhiteSpace($Url)) {
    # Most people arrive here having just copied the URL out of a chat.
    $clip = ''
    try { $clip = Get-Clipboard -Raw } catch { }
    if ($clip -and $clip.Trim() -match '^https?://\S+token=') {
        $suggestion = $clip.Trim()
        Write-Host "  Found a share URL on your clipboard:" -ForegroundColor DarkGray
        Write-Host "  $suggestion" -ForegroundColor DarkGray
        $answer = Read-Host "  Use it? [Y/n]"
        if ($answer -eq '' -or $answer -match '^[Yy]') { $Url = $suggestion }
    }
}

if ([string]::IsNullOrWhiteSpace($Url)) {
    $Url = (Read-Host "  Paste the URL your friend sent you").Trim()
}

if ([string]::IsNullOrWhiteSpace($Url)) {
    Write-Host "  No URL given." -ForegroundColor Red
    exit 1
}

Write-Host ""
& $P2pgpu attach $Url
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Could not connect. Check that:" -ForegroundColor Yellow
    Write-Host "    - Tailscale is running on both machines" -ForegroundColor Yellow
    Write-Host "    - your friend's share has not expired (ask them to run Sharing-Status.bat)" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "  In the notebook, check the GPU is really there:" -ForegroundColor DarkGray
Write-Host "      import torch; print(torch.cuda.get_device_name(0))" -ForegroundColor White
Write-Host ""
Write-Host "  Save anything you want to keep into /workspace." -ForegroundColor DarkGray
Write-Host ""
