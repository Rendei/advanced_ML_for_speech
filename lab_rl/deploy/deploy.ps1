<#
.SYNOPSIS
  Deploys lab_rl to a server over SSH with ONE connection (so the password prompt appears once).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File lab_rl\deploy\deploy.ps1
  powershell -ExecutionPolicy Bypass -File lab_rl\deploy\deploy.ps1 -Server 185.203.6.202 -User root
#>
param(
    [string]$Server = "185.203.6.202",
    [string]$User = "root",
    [string]$Dir = "/opt/c4rl"
)
$ErrorActionPreference = "Stop"
$lab = Resolve-Path (Join-Path $PSScriptRoot "..")
$archive = Join-Path $env:TEMP "c4rl-deploy.tgz"

Write-Host "Packing $lab ..."
tar -czf $archive -C $lab Dockerfile docker-compose.prod.yml pyproject.toml requirements.txt requirements-dev.txt README.md src tests weights deploy
Write-Host ("Archive: {0:N1} MB" -f ((Get-Item $archive).Length / 1MB))

$flags = ""
if ((Read-Host "Install Docker on the server if it is missing? [y/N]") -match '^[yY]') { $flags += "INSTALL_DOCKER=1 " }
Write-Host "Whatever listens on ports 80/443 on the server (the old site: nginx, apache, containers) will be STOPPED and disabled."
if ((Read-Host "Free ports 80/443? [y/N]") -match '^[yY]') { $flags += "FREE_PORTS=1 " }

Write-Host "Connecting to $User@$Server (enter the SSH password when asked) ..."
$remote = "mkdir -p $Dir && tar -xzf - -C $Dir && cd $Dir && $flags sh deploy/remote.sh"
cmd /c "ssh -o StrictHostKeyChecking=accept-new $User@$Server `"$remote`" < `"$archive`""
if ($LASTEXITCODE -ne 0) { throw "deployment failed (exit code $LASTEXITCODE)" }
Remove-Item $archive
Write-Host "Done: https://rendei-cool.ru"
