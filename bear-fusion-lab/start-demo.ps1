param([int]$Port = 8765, [int]$Workers = 1)
$ErrorActionPreference = 'Stop'
$demoPython = Join-Path $PSScriptRoot '../bear-text-beta/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $demoPython)) { throw 'Python окружение не найдено. См. DEMO.md.' }
if (-not $env:BEAR_TESSERACT -and (Test-Path -LiteralPath 'C:/msys64/ucrt64/bin/tesseract.exe')) {
    $env:BEAR_TESSERACT = 'C:/msys64/ucrt64/bin/tesseract.exe'
}
& $demoPython (Join-Path $PSScriptRoot 'demo.py') --port $Port --workers $Workers
