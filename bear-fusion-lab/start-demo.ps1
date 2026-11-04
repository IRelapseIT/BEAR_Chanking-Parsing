param(
    [int]$Port = 8765,
    [int]$Workers = 1,
    [string]$Python = "python",
    [Parameter(Mandatory=$true)][string]$Upstream,
    [Parameter(Mandatory=$true)][string]$ParserPython,
    [Parameter(Mandatory=$true)][string]$Tokenizer
)
$ErrorActionPreference = 'Stop'
& $Python (Join-Path $PSScriptRoot 'demo.py') --port $Port --workers $Workers `
    --upstream $Upstream --parser-python $ParserPython --tokenizer $Tokenizer
exit $LASTEXITCODE
