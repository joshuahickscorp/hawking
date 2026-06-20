# make-shim.ps1 — build the windowless .sa double-click shim on Windows.
#
# Produces strand-open.exe in the output dir. Ship it alongside strand.exe;
# the shim re-invokes the CLI with no console flash. Pair with the per-user
# association, which `strand register` writes to point .sa at this shim when
# it is present (falls back to strand.exe directly otherwise).
#
# Usage:  .\make-shim.ps1 [-OutDir <dir>]
param(
    [string]$OutDir = "$PSScriptRoot\build"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$src = Join-Path $PSScriptRoot "strand-open.rs"
$out = Join-Path $OutDir "strand-open.exe"

Write-Host "-> compiling windowless shim (GUI subsystem)"
& rustc -O --edition 2021 -C link-args=user32.lib `
    --crate-name strand_open $src -o $out
if ($LASTEXITCODE -ne 0) { throw "rustc failed ($LASTEXITCODE)" }

Write-Host "OK: $out"
Write-Host "Ship strand-open.exe next to strand.exe, then run: strand register"
