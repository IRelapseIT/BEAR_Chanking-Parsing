param(
    [string]$Python,
    [string]$ModelDirectory,
    [int]$Port = 7998,
    [int]$Threads = 4,
    [int]$Microbatch = 4
)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $PSScriptRoot '.infinity-venv/Scripts/python.exe' }
if (-not $ModelDirectory) { $ModelDirectory = Join-Path $PSScriptRoot 'models/bge-m3' }
& $Python -I -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 'Python 3.12 required')"
if ($LASTEXITCODE) { exit $LASTEXITCODE }

# Stand startup uses scripts/services.py and actual Infinity. This is the legacy standalone compatibility server.
$bgeCore = Join-Path $PSScriptRoot 'bear-text-beta/src'
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $bgeCore + [IO.Path]::PathSeparator + $previousPythonPath
    & $Python -m bear_text_beta.local_inference --model-dir $ModelDirectory `
        --port $Port --threads $Threads --microbatch $Microbatch
    $exitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}
exit $exitCode
