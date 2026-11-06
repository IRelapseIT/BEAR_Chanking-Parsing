param([string]$Python = "python", [switch]$DownloadQdrant)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 required'"
if ($LASTEXITCODE) { throw 'Python 3.12 required' }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { & $Python -m venv .venv }
& ./.venv/Scripts/python.exe -m pip install -r requirements-workflow-api.txt
if ($LASTEXITCODE) { throw 'API dependencies failed' }
& ./.venv/Scripts/python.exe -m pip install --no-deps -e './bear-text-beta' -e './bear-fusion-lab'
if ($LASTEXITCODE) { throw 'Local packages failed' }
if (-not (Test-Path -LiteralPath '.infinity-venv/Scripts/python.exe')) { & $Python -m venv .infinity-venv }
& ./.infinity-venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
if ($LASTEXITCODE) { throw 'CPU PyTorch failed' }
& ./.infinity-venv/Scripts/python.exe -m pip install -r requirements-workflow-infinity.txt
if ($LASTEXITCODE) { throw 'Infinity dependencies failed' }
if ($DownloadQdrant) {
    New-Item -ItemType Directory -Force runtime/bin | Out-Null
    $archive = Join-Path $PSScriptRoot 'runtime/qdrant-1.20.0.zip'
    Invoke-WebRequest -Uri 'https://github.com/qdrant/qdrant/releases/download/v1.20.0/qdrant-x86_64-pc-windows-msvc.zip' -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne '6d13553584fd0411f00637519e293331ff0aa1fb79bd9e30db45839bb75de2be') { throw 'Qdrant archive checksum mismatch' }
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $PSScriptRoot 'runtime/bin') -Force
}
& ./.venv/Scripts/bear.exe --model-dir models/bge-m3 doctor --require-model
if ($LASTEXITCODE) { throw 'Prepare the pinned BGE-M3 model/tokenizer before startup' }
