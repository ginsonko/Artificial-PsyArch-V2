$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$DefaultPort = 8766
$OutputsDir = Join-Path $RepoRoot "outputs"
$PidFile = Join-Path $OutputsDir ("observatory_{0}.pid" -f $DefaultPort)
$StdoutLog = Join-Path $OutputsDir "observatory_restart_stdout.log"
$StderrLog = Join-Path $OutputsDir "observatory_restart_stderr.log"

New-Item -ItemType Directory -Force -Path $OutputsDir | Out-Null

function Resolve-PythonCommand {
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return @($venvPython)
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @("py", "-3")
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }
    throw "Python 3.11+ was not found. Run install first."
}

function Stop-ProcessIfAlive([int]$ProcessId) {
    if ($ProcessId -le 0) { return }
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        try {
            Stop-Process -Id $ProcessId -Force -ErrorAction Stop
            Start-Sleep -Milliseconds 600
        } catch {
        }
    }
}

function Get-ListeningPids([int]$Port) {
    $rows = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($null -eq $rows) { return @() }
    return @($rows | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Remove-StalePortListeners([int]$Port, [int[]]$KnownPids) {
    $candidates = @()
    if ($KnownPids) {
        $candidates += $KnownPids
    }
    $candidates += Get-ListeningPids $Port
    $targets = @($candidates | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    foreach ($target in $targets) {
        Write-Host ("Stopping stale listener PID={0} on port {1}..." -f $target, $Port)
        Stop-ProcessIfAlive ([int]$target)
    }
}

$pyCmd = Resolve-PythonCommand

Write-Host "======================================"
Write-Host "       AP Phase 2 Observatory V2"
Write-Host "======================================"
Write-Host ("Repo root: {0}" -f $RepoRoot)
Write-Host ("Command : {0} -m observatory_v2 serve --host 127.0.0.1 --port {1} --no-browser" -f ($pyCmd -join " "), $DefaultPort)
Write-Host ""

$knownPids = @()
if (Test-Path $PidFile) {
    $oldPidText = (Get-Content -LiteralPath $PidFile -Encoding UTF8 -ErrorAction SilentlyContinue | Select-Object -First 1)
    $oldPid = 0
    [void][int]::TryParse(($oldPidText | Out-String).Trim(), [ref]$oldPid)
    if ($oldPid -gt 0) {
        $knownPids += $oldPid
        Write-Host ("Found previous recorded PID={0}" -f $oldPid)
    }
}

Remove-StalePortListeners -Port $DefaultPort -KnownPids $knownPids

if (Test-Path $StdoutLog) { Remove-Item -LiteralPath $StdoutLog -Force -ErrorAction SilentlyContinue }
if (Test-Path $StderrLog) { Remove-Item -LiteralPath $StderrLog -Force -ErrorAction SilentlyContinue }

$argList = @()
if ($pyCmd.Count -gt 1) {
    $argList += $pyCmd[1..($pyCmd.Count - 1)]
}
$argList += @("-m", "observatory_v2", "serve", "--host", "127.0.0.1", "--port", "$DefaultPort", "--no-browser")

$proc = Start-Process -FilePath $pyCmd[0] -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -WindowStyle Hidden -PassThru

$startedPid = 0
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    $pids = Get-ListeningPids $DefaultPort
    if ($pids.Count -gt 0) {
        $startedPid = [int]$pids[0]
        break
    }
    if ($proc.HasExited) {
        break
    }
}

if ($startedPid -le 0) {
    Write-Host ("[ERROR] Observatory did not start on port {0}" -f $DefaultPort)
    if (Test-Path $StderrLog) {
        Write-Host "---------- stderr ----------"
        Get-Content -LiteralPath $StderrLog -Encoding UTF8
        Write-Host "----------------------------"
    }
    exit 1
}

Set-Content -LiteralPath $PidFile -Value $startedPid -Encoding UTF8
Write-Host ("Observatory started. PID={0}" -f $startedPid)
Write-Host ("URL: http://127.0.0.1:{0}/" -f $DefaultPort)
Write-Host ""
Write-Host "Logs:"
Write-Host "  outputs\\observatory_restart_stdout.log"
Write-Host "  outputs\\observatory_restart_stderr.log"

Start-Process ("http://127.0.0.1:{0}/" -f $DefaultPort) | Out-Null
