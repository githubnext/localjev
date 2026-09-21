#Requires -Version 5.1
<#
.SYNOPSIS
Start the native ROCm model server and LocalJev on Windows.
.DESCRIPTION
Uses an already prepared Python environment and model cache. The backend is
started in a hidden process, while LocalJev runs in the current terminal.
Press Ctrl+C to stop both. Existing listeners are never stopped.
Automatic Bun .env loading is disabled; parameters override inherited LocalJev
connection settings. LOCALJEV_API_KEY and LOCALJEV_UPSTREAM_API_KEY are inherited
unless their corresponding parameters are supplied.
.EXAMPLE
pwsh -File scripts/start-rocm.ps1 -LocalFilesOnly
.EXAMPLE
pwsh -File scripts/start-rocm.ps1 -PythonPath C:\Python313\python.exe -CheckOnly
#>
[CmdletBinding()]
param(
    [Alias('Python')]
    [ValidateNotNullOrEmpty()]
    [string] $PythonPath = '.venv-rocm\Scripts\python.exe',
    [string] $Bun,
    [ValidateNotNullOrEmpty()]
    [string] $Model = 'google/diffusiongemma-26B-A4B-it',
    [string] $Revision,
    [ValidateRange(1, 65535)]
    [int] $BackendPort = 8000,
    [ValidateRange(1, 65535)]
    [int] $Port = 8080,
    [System.Net.IPAddress] $ListenAddress = [System.Net.IPAddress]::Loopback,
    [ValidateRange(0, 2147483647)]
    [int] $Device = 0,
    [ValidateSet('float16', 'bfloat16', 'float32')]
    [string] $Dtype = 'float16',
    [ValidateRange(1, 2147483647)]
    [int] $MaxInputTokens = 8192,
    [ValidateRange(1, 2147483647)]
    [int] $MaxOutputTokens = 2048,
    [ValidateRange(1, 86400)]
    [int] $StartupTimeoutSeconds = 1200,
    [ValidateRange(1, 86400)]
    [int] $TimeoutSeconds = 300,
    [string] $ApiKey = $env:LOCALJEV_API_KEY,
    [string] $UpstreamApiKey = $env:LOCALJEV_UPSTREAM_API_KEY,
    [switch] $LocalFilesOnly,
    [switch] $CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-Executable([string] $Name) {
    if ([System.IO.Path]::IsPathRooted($Name) -or $Name.Contains('\') -or $Name.Contains('/')) {
        $candidate = if ([System.IO.Path]::IsPathRooted($Name)) { $Name } else { Join-Path $repoRoot $Name }
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Executable not found: $candidate. Prepare the ROCm environment first or specify -PythonPath."
        }
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    $command = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Executable '$Name' was not found on PATH. Install the required dependency first."
    }
    return @($command)[0].Source
}

function Assert-PortAvailable([System.Net.IPAddress] $Address, [int] $Number) {
    $listener = [System.Net.Sockets.TcpListener]::new($Address, $Number)
    try {
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
    }
    catch {
        throw "Cannot bind ${Address}:${Number}. Choose another port or stop the existing listener yourself. $($_.Exception.Message)"
    }
    finally {
        $listener.Stop()
    }
}

function ConvertTo-WindowsArgument([string] $Argument) {
    # Start-Process joins ArgumentList before passing it to CreateProcess.
    # Quote using the Windows C-runtime rules, including trailing backslashes.
    $escaped = [regex]::Replace($Argument, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw 'This launcher requires Windows. Run scripts/rocm_server.py directly on other platforms.'
}
if ($BackendPort -eq $Port) {
    throw '-BackendPort and -Port must be different.'
}
if ([string]::IsNullOrWhiteSpace($Model)) {
    throw '-Model must not be blank.'
}

$python = Resolve-Executable $PythonPath
$bundledBun = Join-Path $repoRoot '.runtime\bun\bun-windows-x64\bun.exe'
if (-not $Bun) {
    $Bun = if (Test-Path -LiteralPath $bundledBun -PathType Leaf) { $bundledBun } else { 'bun' }
}
$bunExecutable = Resolve-Executable $Bun
$backendScript = Join-Path $PSScriptRoot 'rocm_server.py'
$bridgeScript = Join-Path $repoRoot 'src\index.ts'
foreach ($requiredFile in @($backendScript, $bridgeScript)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required source file not found: $requiredFile"
    }
}

Assert-PortAvailable ([System.Net.IPAddress]::Loopback) $BackendPort
Assert-PortAvailable $ListenAddress $Port

$backendProcess = $null
$savedEnvironment = @{}
$locationPushed = $false
try {
    Push-Location -LiteralPath $repoRoot
    $locationPushed = $true
    $bunHelp = & $bunExecutable --help 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $bunHelp -notmatch '--no-env-file') {
        throw 'This launcher requires a Bun version with --no-env-file support. Update Bun first.'
    }

    # Fail before launching either server when the selected Python is incomplete
    # or silently resolves to CPU/CUDA PyTorch instead of ROCm.
    $preflight = @'
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
required = ("torch", "transformers", "accelerate", "PIL")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python dependencies: " + ", ".join(missing))
from packaging.version import Version
transformers_version = version("transformers")
transformers_root = Path(importlib.util.find_spec("transformers").origin).parent
diffusion_module = transformers_root / "models" / "diffusion_gemma" / "modeling_diffusion_gemma.py"
if Version(transformers_version) < Version("5.11.0") or not diffusion_module.is_file():
    raise SystemExit("Transformers 5.11.0 or newer with DiffusionGemma support is required.")
if "class DiffusionGemmaForBlockDiffusion(" not in diffusion_module.read_text(encoding="utf-8"):
    raise SystemExit("The selected Transformers build has no DiffusionGemmaForBlockDiffusion model.")
import torch
if not torch.version.hip:
    raise SystemExit("The selected Python does not have a ROCm PyTorch build.")
device = int(sys.argv[1])
if not torch.cuda.is_available() or device >= torch.cuda.device_count():
    raise SystemExit("The selected ROCm GPU device is not available: " + str(device))
props = torch.cuda.get_device_properties(device)
architecture = str(getattr(props, "gcnArchName", "unknown architecture"))
if architecture.split(":")[0] != "gfx1151":
    raise SystemExit("This launcher targets gfx1151; selected GPU architecture: " + architecture)
print("Python: " + sys.executable)
print("ROCm: " + str(torch.version.hip))
print("Transformers: " + transformers_version)
print("GPU: " + props.name + " (" + architecture + ")")
'@
    # Stdin avoids PowerShell 5.1's legacy native argument handling stripping
    # embedded quotes from a multiline Python -c argument.
    $preflight | & $python - $Device
    if ($LASTEXITCODE -ne 0) {
        throw 'ROCm preflight failed. See the dependency/GPU error above.'
    }
    if ($CheckOnly) {
        Write-Host 'Launcher checks passed. No model was loaded and no server was started.'
        return
    }

    $bridgeEnvironment = @{
        LOCALJEV_UPSTREAM = "http://127.0.0.1:$BackendPort"
        LOCALJEV_UPSTREAM_MODEL = $Model
        LOCALJEV_HOST = $ListenAddress.ToString()
        LOCALJEV_PORT = "$Port"
        LOCALJEV_MAX_INFLIGHT = '1'
        LOCALJEV_TIMEOUT = "$TimeoutSeconds"
        LOCALJEV_MAX_OUTPUT_TOKENS = "$MaxOutputTokens"
        LOCALJEV_API_KEY = $ApiKey
        LOCALJEV_UPSTREAM_API_KEY = $UpstreamApiKey
    }
    foreach ($entry in $bridgeEnvironment.GetEnumerator()) {
        $savedEnvironment[$entry.Key] = [System.Environment]::GetEnvironmentVariable($entry.Key, 'Process')
        [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    $healthHeaders = @{}
    if ($UpstreamApiKey) { $healthHeaders.Authorization = "Bearer $UpstreamApiKey" }

    $runtimeDirectory = Join-Path $repoRoot '.runtime'
    $null = New-Item -ItemType Directory -Path $runtimeDirectory -Force
    $runId = '{0}-{1}-{2}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID, ([guid]::NewGuid().ToString('N').Substring(0, 8))
    $stdoutLog = Join-Path $runtimeDirectory "rocm-$runId.stdout.log"
    $stderrLog = Join-Path $runtimeDirectory "rocm-$runId.stderr.log"
    $backendArguments = @(
        '-u', $backendScript,
        '--model', $Model,
        '--host', '127.0.0.1',
        '--port', "$BackendPort",
        '--device', "$Device",
        '--dtype', $Dtype,
        '--max-input-tokens', "$MaxInputTokens",
        '--max-output-tokens', "$MaxOutputTokens",
        '--max-queue', '2'
    )
    if ($LocalFilesOnly) { $backendArguments += '--local-files-only' }
    if ($Revision) { $backendArguments += @('--revision', $Revision) }
    $argumentLine = ($backendArguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
    $backendProcess = Start-Process -FilePath $python -ArgumentList $argumentLine -WorkingDirectory $repoRoot `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
    $null = $backendProcess.Handle
    Write-Host "ROCm backend PID $($backendProcess.Id); logs: $stdoutLog and $stderrLog"
    Write-Host "Waiting up to $StartupTimeoutSeconds seconds for model '$Model'..."

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $healthy = $false
    while ($timer.Elapsed.TotalSeconds -lt $StartupTimeoutSeconds) {
        $backendProcess.Refresh()
        if ($backendProcess.HasExited) {
            throw "ROCm backend exited with code $($backendProcess.ExitCode). Inspect $stderrLog and $stdoutLog."
        }
        try {
            $remainingSeconds = [Math]::Max(1, [Math]::Ceiling($StartupTimeoutSeconds - $timer.Elapsed.TotalSeconds))
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -Headers $healthHeaders -TimeoutSec ([Math]::Min(5, $remainingSeconds))
            if ($health.status -eq 'ok' -and $health.model -eq $Model) {
                $healthy = $true
                break
            }
        }
        catch {
            # The backend can be unreachable until the model finishes loading.
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $healthy) {
        throw "ROCm model startup timed out after $StartupTimeoutSeconds seconds. Inspect $stderrLog and $stdoutLog."
    }
    $backendProcess.Refresh()
    if ($backendProcess.HasExited) {
        throw "ROCm backend exited after its health check. Inspect $stderrLog and $stdoutLog."
    }

    Write-Host "Starting LocalJev at http://${ListenAddress}:$Port; press Ctrl+C to stop both servers."
    & $bunExecutable --no-env-file run $bridgeScript
    if ($LASTEXITCODE -ne 0) {
        throw "LocalJev exited with code $LASTEXITCODE."
    }
}
finally {
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    if ($null -ne $backendProcess) {
        $backendProcess.Refresh()
        if (-not $backendProcess.HasExited) {
            # Windows venv python.exe can launch the actual interpreter as a
            # child. Stop the owned tree so that worker cannot outlive LocalJev.
            try {
                if ($PSVersionTable.PSVersion.Major -ge 7) {
                    $backendProcess.Kill($true)
                }
                else {
                    & "$env:SystemRoot\System32\taskkill.exe" /PID $backendProcess.Id /T /F | Out-Null
                }
            }
            catch [System.InvalidOperationException] {
                # The process can exit between HasExited and Kill.
            }
            if (-not $backendProcess.WaitForExit(10000)) {
                Write-Warning "Unable to stop the owned ROCm backend tree (PID $($backendProcess.Id))."
            }
        }
        $backendProcess.Dispose()
    }
    if ($locationPushed) { Pop-Location }
}
