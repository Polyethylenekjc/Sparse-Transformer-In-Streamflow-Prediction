param(
    [ValidateSet('prepare', 'run', 'summarize')][string]$Action = 'prepare',
    [string]$Python = 'python',
    [string]$Config = 'configs/revision_minimal.json',
    [string]$DataRoot = '',
    [string]$OutputRoot = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$taskArguments = @('run_revision.py', $Action, '--config', $Config)
if ($DataRoot) { $taskArguments += @('--data-root', $DataRoot) }
if ($OutputRoot) { $taskArguments += @('--output-root', $OutputRoot) }
& $Python @taskArguments
exit $LASTEXITCODE
