# Dot-source before launching API/CLI: . ./use-bge-m3.ps1
$env:BEAR_MODEL_DIR = Join-Path $PSScriptRoot 'models/bge-m3'
$env:BEAR_STATE_DIR = Join-Path $PSScriptRoot 'state-bge-m3'
$env:BEAR_INFINITY_URL = 'http://127.0.0.1:7998/v1'
$env:BEAR_INFINITY_MODEL = 'BAAI/bge-m3'
$env:BEAR_QDRANT_URL = 'http://127.0.0.1:6335'
# Profile-specific collection selection is implemented in Services.
Remove-Item Env:BEAR_QDRANT_COLLECTION -ErrorAction SilentlyContinue
Remove-Item Env:BEAR_TEST_COUNTER -ErrorAction SilentlyContinue
Remove-Item Env:BEAR_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:BEAR_CHUNKER_URL -ErrorAction SilentlyContinue
$env:PYTHONPATH = (Join-Path $PSScriptRoot 'bear-text-beta/src') + ';' + (Join-Path $PSScriptRoot 'bear-fusion-lab/src')
