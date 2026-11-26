param(
    [string]$Python = 'python',
    [switch]$DownloadQdrant,
    [switch]$PrepareModels,
    [switch]$WithPaddle,
    [switch]$WithDocling,
    [string]$Tesseract,
    [string]$LibreOffice,
    [switch]$Start
)

$ErrorActionPreference = 'Stop'

function Invoke-BearCommand([string]$Executable, [string[]]$CommandArgs) {
    & $Executable @CommandArgs
    if ($LASTEXITCODE) { throw "Failed ($LASTEXITCODE): $Executable $($CommandArgs -join ' ')" }
}

function Confirm-Python312([string]$Executable) {
    Invoke-BearCommand $Executable @('-I', '-c', "import sys; print(sys.executable, sys.version.split()[0]); sys.exit(0 if sys.version_info[:2] == (3, 12) else 'Python 3.12 required')")
}

function Get-BearVenv([string]$Directory) {
    $executable = Join-Path $Directory 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $executable)) {
        Invoke-BearCommand $Python @('-m', 'venv', $Directory) | Write-Host
    }
    Confirm-Python312 $executable | Write-Host
    return (Resolve-Path -LiteralPath $executable).Path
}

Push-Location -LiteralPath $PSScriptRoot
try {
    Confirm-Python312 $Python
    $api = Get-BearVenv '.venv'
    $infinity = Get-BearVenv '.infinity-venv'
    Invoke-BearCommand $api @('-m', 'pip', 'install', '-r', 'requirements-workflow-api.txt')
    Invoke-BearCommand $api @('-m', 'pip', 'install', '--no-deps', '-e', './bear-text-beta', '-e', './bear-fusion-lab')
    Invoke-BearCommand $api @('-m', 'pip', 'check')
    Invoke-BearCommand $infinity @('-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.8.0')
    Invoke-BearCommand $infinity @('-m', 'pip', 'install', '-r', 'requirements-workflow-infinity.txt')
    Invoke-BearCommand $infinity @('-m', 'pip', 'check')

    $localConfig = [ordered]@{}
    if ($WithPaddle) {
        $paddle = Get-BearVenv '.ocr-venv'
        Invoke-BearCommand $paddle @('-m', 'pip', 'install', '-r', 'requirements-paddle-ocr.txt')
        Invoke-BearCommand $paddle @('-m', 'pip', 'check')
        $localConfig['paddle_python'] = '.ocr-venv/Scripts/python.exe'
        $localConfig['paddle_models'] = 'models/paddle-ocr'
        $localConfig['paddle_layout_model'] = 'models/paddle-ocr/PP-DocLayout_plus-L'
    }
    if ($WithDocling) {
        $docling = Get-BearVenv 'runtime/venvs/docling'
        Invoke-BearCommand $docling @('-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.8.0')
        Invoke-BearCommand $docling @('-m', 'pip', 'install', 'docling==2.148.0', 'docling-core==2.106.0')
        Invoke-BearCommand $docling @('-m', 'pip', 'check')
        $localConfig['docling_python'] = 'runtime/venvs/docling/Scripts/python.exe'
        $localConfig['docling_artifacts'] = 'models/docling'
    }
    if ($Tesseract) { $localConfig['tesseract_exe'] = (Resolve-Path -LiteralPath $Tesseract).Path }
    if ($LibreOffice) { $localConfig['libreoffice_exe'] = (Resolve-Path -LiteralPath $LibreOffice).Path }

    if ($DownloadQdrant -and -not (Test-Path -LiteralPath 'runtime/bin/qdrant.exe')) {
        New-Item -ItemType Directory -Force runtime/bin | Out-Null
        $archive = Join-Path $PSScriptRoot 'runtime/qdrant-1.20.0.zip'
        Invoke-WebRequest -Uri 'https://github.com/qdrant/qdrant/releases/download/v1.20.0/qdrant-x86_64-pc-windows-msvc.zip' -OutFile $archive
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne '6d13553584fd0411f00637519e293331ff0aa1fb79bd9e30db45839bb75de2be') {
            throw 'Qdrant archive checksum mismatch'
        }
        Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $PSScriptRoot 'runtime/bin') -Force
    }

    if ($PrepareModels) {
        $modelArgs = @('scripts/prepare_stand_models.py')
        if ($WithPaddle) { $modelArgs += '--with-paddle' }
        if ($WithDocling) { $modelArgs += '--with-docling' }
        Invoke-BearCommand $api $modelArgs
    }

    $configPath = 'runtime/stand-local.json'
    if ($localConfig.Count -gt 0 -and -not (Test-Path -LiteralPath $configPath)) {
        New-Item -ItemType Directory -Force runtime | Out-Null
        $localConfig | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath $configPath
    } elseif ((Test-Path -LiteralPath $configPath) -and $localConfig.Count -gt 0) {
        Write-Host "Existing configuration kept: $configPath"
    }
    $doctorArgs = @('scripts/services.py', 'doctor')
    $startArgs = @('scripts/services.py', 'start')
    if (Test-Path -LiteralPath $configPath) {
        $doctorArgs += @('--config', $configPath)
        $startArgs += @('--config', $configPath)
    }
    Invoke-BearCommand $api $doctorArgs
    if ($Start) { Invoke-BearCommand $api $startArgs }
}
finally {
    Pop-Location
}
