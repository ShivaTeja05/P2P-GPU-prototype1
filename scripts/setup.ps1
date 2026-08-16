<#
.SYNOPSIS
    One-time setup for p2pgpu on Windows.

.DESCRIPTION
    Checks for Python, uv, Tailscale and (if you plan to share a GPU) Docker
    Desktop. Offers to install anything missing via winget, then installs
    p2pgpu into a local virtual environment.

    Safe to re-run. Nothing is installed without asking first.

.PARAMETER GuestOnly
    You only want to USE a friend's GPU, not share yours. Skips Docker.

.PARAMETER Yes
    Install missing prerequisites without prompting.
#>
[CmdletBinding()]
param(
    [switch]$GuestOnly,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [ok] $msg"   -ForegroundColor Green }
function Write-Miss($msg) { Write-Host "  [--] $msg"   -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "  [!!] $msg"   -ForegroundColor Red }

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# Tailscale's installer does not add itself to PATH, so look where it lands.
function Find-Tailscale {
    if (Test-Command 'tailscale') { return (Get-Command tailscale).Source }
    foreach ($p in @(
        "$env:ProgramFiles\Tailscale\tailscale.exe",
        "${env:ProgramFiles(x86)}\Tailscale\tailscale.exe"
    )) { if (Test-Path $p) { return $p } }
    return $null
}

# A freshly installed program is not on this session's PATH -- the installer
# updates the registry, but the environment block of an already-running process
# is a snapshot taken at launch. Without this, installing Python or uv and then
# using it in the same run fails with "not recognized", and the user is told to
# close and reopen the window. Re-reading the registry avoids that entirely.
function Update-SessionPath {
    $machine = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $extra   = "$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:USERPROFILE\.local\bin"
    $env:Path = (@($machine, $user, $extra) | Where-Object { $_ }) -join ';'
}

function Invoke-Winget($id, $label) {
    if (-not (Test-Command 'winget')) {
        Write-Bad "winget is not available. Install $label manually."
        return $false
    }
    Write-Host "  installing $label ..." -ForegroundColor DarkGray
    winget install --id $id --accept-source-agreements --accept-package-agreements -h
    if ($LASTEXITCODE -ne 0) {
        Write-Bad "winget could not install $label (exit $LASTEXITCODE). Install it manually."
        return $false
    }
    Update-SessionPath
    Write-Ok "$label installed"
    return $true
}

function Confirm-Install($label) {
    if ($Yes) { return $true }
    $answer = Read-Host "  Install $label now? [Y/n]"
    return ($answer -eq '' -or $answer -match '^[Yy]')
}

Write-Host @"

  p2pgpu - Windows setup
  ----------------------
  Role: $(if ($GuestOnly) { 'guest (use a friend''s GPU)' } else { 'host (share this PC''s GPU)' })

"@ -ForegroundColor White

# --------------------------------------------------------------------------
Write-Step 'Checking Windows version'
$build = [System.Environment]::OSVersion.Version.Build
if ($build -lt 19041) {
    Write-Bad "Windows build $build is too old. WSL2 and Docker Desktop need build 19041 (Windows 10 2004) or newer."
    if (-not $GuestOnly) { exit 1 }
} else {
    Write-Ok "Windows build $build"
}

# --------------------------------------------------------------------------
Write-Step 'Checking Python 3.10-3.12'
$pythonOk = $false
foreach ($candidate in @('python3.12', 'python3.11', 'python3.10', 'python')) {
    if (-not (Test-Command $candidate)) { continue }
    $verText = & $candidate --version 2>&1
    if ($verText -match '(\d+)\.(\d+)\.\d+') {
        $major = [int]$Matches[1]; $minor = [int]$Matches[2]
        if ($major -eq 3 -and $minor -ge 10 -and $minor -le 12) {
            Write-Ok "$verText ($candidate)"
            $pythonOk = $true
            break
        }
    }
}
if (-not $pythonOk) {
    Write-Miss 'No suitable Python found (need 3.10-3.12; PyTorch has no 3.13+ wheels yet)'
    Write-Host '  Not fatal: uv can download its own Python if needed.' -ForegroundColor DarkGray
    if (Confirm-Install 'Python 3.12') {
        Invoke-Winget 'Python.Python.3.12' 'Python 3.12' | Out-Null
    }
}

# --------------------------------------------------------------------------
Write-Step 'Checking uv'
if (Test-Command 'uv') {
    Write-Ok 'uv'
} else {
    Write-Miss 'uv not found (fast Python package manager)'
    if (Confirm-Install 'uv') {
        if (-not (Invoke-Winget 'astral-sh.uv' 'uv')) {
            Write-Host '  falling back to the official installer ...' -ForegroundColor DarkGray
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        }
        Update-SessionPath
    }
}

# --------------------------------------------------------------------------
Write-Step 'Checking Tailscale'
$ts = Find-Tailscale
if ($ts) {
    Write-Ok "Tailscale ($ts)"
    $tsIp = (& $ts ip -4 2>$null | Select-Object -First 1)
    if ($tsIp) {
        Write-Ok "Tailscale IP: $tsIp"
    } else {
        Write-Miss "Tailscale installed but not connected. Run: `"$ts`" up"
    }
} else {
    Write-Miss 'Tailscale not found'
    if (Confirm-Install 'Tailscale') {
        Invoke-Winget 'tailscale.tailscale' 'Tailscale' | Out-Null
        Write-Host '  After install: sign in with YOUR OWN account (no need to share a login).' -ForegroundColor DarkGray
    }
}

# --------------------------------------------------------------------------
if (-not $GuestOnly) {
    Write-Step 'Checking Docker Desktop (needed to share a GPU)'
    if (Test-Command 'docker') {
        docker info *>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok 'Docker is installed and running'
        } else {
            Write-Miss 'Docker is installed but not running. Start Docker Desktop, then re-run this script.'
        }
    } else {
        Write-Miss 'Docker Desktop not found'
        if (Confirm-Install 'Docker Desktop') {
            Invoke-Winget 'Docker.DockerDesktop' 'Docker Desktop' | Out-Null
            Write-Host ''
            Write-Host '  IMPORTANT, after Docker Desktop installs:' -ForegroundColor Yellow
            Write-Host '    1. Reboot if it asks.' -ForegroundColor Yellow
            Write-Host '    2. Start Docker Desktop and finish first-run setup.' -ForegroundColor Yellow
            Write-Host '    3. Settings > General > enable "Use the WSL 2 based engine".' -ForegroundColor Yellow
            Write-Host '    4. Keep your NVIDIA *Windows* driver up to date.' -ForegroundColor Yellow
            Write-Host '       Do NOT install Linux NVIDIA drivers inside WSL - that breaks it.' -ForegroundColor Yellow
            Write-Host ''
        }
    }

    Write-Step 'Checking NVIDIA GPU'
    if (Test-Command 'nvidia-smi') {
        $gpuLines = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
        if ($gpuLines) {
            foreach ($line in $gpuLines) { Write-Ok $line }
        } else {
            Write-Miss 'nvidia-smi ran but reported no GPU'
        }
    } else {
        Write-Miss 'nvidia-smi not found - is an NVIDIA driver installed?'
    }
}

# --------------------------------------------------------------------------
Write-Step 'Installing p2pgpu'
Push-Location $RepoRoot
try {
    if (Test-Command 'uv') {
        # Re-running Setup.bat is normal (reboot after Docker, retry after a
        # fix). 'uv venv' prompts interactively when .venv already exists, which
        # stalls the script behind a question most people won't expect, so
        # reuse the existing environment instead of recreating it.
        if (Test-Path (Join-Path $RepoRoot '.venv')) {
            Write-Ok 'virtual environment already exists, reusing it'
        } else {
            uv venv --python 3.12
        }
        uv pip install -e .
    } else {
        Write-Miss 'uv unavailable; falling back to venv + pip'
        if (-not (Test-Path (Join-Path $RepoRoot '.venv'))) { python -m venv .venv }
        & '.venv\Scripts\python.exe' -m pip install --quiet --upgrade pip
        & '.venv\Scripts\python.exe' -m pip install -e .
    }
    Write-Ok 'p2pgpu installed'
} catch {
    Write-Bad "Install failed: $_"
    exit 1
} finally {
    Pop-Location
}

# --------------------------------------------------------------------------
Write-Step 'Done'
Write-Host ''
if ($GuestOnly) {
    Write-Host '  To connect to your friend''s GPU, double-click:' -ForegroundColor White
    Write-Host '      Connect-To-GPU.bat' -ForegroundColor Green
} else {
    Write-Host '  Next: verify this PC can share its GPU -' -ForegroundColor White
    Write-Host '      Check-Setup.bat' -ForegroundColor Green
    Write-Host ''
    Write-Host '  Then, to share it:' -ForegroundColor White
    Write-Host '      Share-My-GPU.bat' -ForegroundColor Green
}
Write-Host ''
