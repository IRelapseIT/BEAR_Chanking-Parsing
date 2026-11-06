param(
    [string]$Python = "python",
    [int]$Port = 7998,
    [int]$Threads = 4,
    [int]$Microbatch = 4
)
$ErrorActionPreference = 'Stop'
$bgeCore = Join-Path $PSScriptRoot 'bear-text-beta/src'
$env:PYTHONPATH = $bgeCore + ';' + $env:PYTHONPATH
& $Python -m bear_text_beta.local_inference --model-dir (Join-Path $PSScriptRoot 'models/bge-m3') `
    --port $Port --threads $Threads --microbatch $Microbatch
exit $LASTEXITCODE
