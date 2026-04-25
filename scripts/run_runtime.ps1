# Run the Clash Royale bot runtime from the repository root (PowerShell).
Set-Location (Join-Path $PSScriptRoot "..")
python (Join-Path $PSScriptRoot "runtime/run_runtime.py") @args
exit $LASTEXITCODE
